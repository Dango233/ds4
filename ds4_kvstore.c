#include "ds4_kvstore.h"

/* Shared disk KV checkpoint file support.
 *
 * The low-level file layout and payload helpers are intentionally shared.  The
 * ds4-server still owns the automatic byte-prefix cache policy built on top of
 * this file; ds4-agent uses only the same durable format for explicit sessions,
 * with its own policy in ds4_agent.c.  Protocol-specific extras, such as the
 * server's tool-id -> exact DSML trailer, are attached through trailer hooks and
 * still live with the protocol code that owns those mappings. */

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define KV_CACHE_MAGIC0 'K'
#define KV_CACHE_MAGIC1 'V'
#define KV_CACHE_MAGIC2 'C'
#define KV_CACHE_VERSION 1u
/* Header byte 20 carries the graph-payload ABI.  It is separate from the outer
 * file version because the KVC envelope can remain stable while the serialized
 * ds4_session internals become unsafe to restore across runtime changes. */
#define KV_CACHE_PAYLOAD_ABI 2u
#define KV_CACHE_DEFAULT_MIN_TOKENS 512
#define KV_CACHE_DEFAULT_COLD_MAX_TOKENS 30000
/* Tokenizers may merge text across the prompt boundary. Trimming a small tail
 * still improves the cheap token-prefix path, while text-prefix lookup handles
 * cases where canonical prompt tokenization spells the same bytes differently.
 * The 2048 alignment also matches the backend prefill chunk schedule, which
 * keeps compressor row finalization identical to a cold full prompt. */
#define KV_CACHE_DEFAULT_BOUNDARY_TRIM_TOKENS 32
#define KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS 2048
#define KV_CACHE_DEFAULT_CONTINUED_INTERVAL_TOKENS 10000
/* Disk-hit counts are evidence that a checkpoint was useful, but only while
 * the workload still resembles the one that produced those hits. */
#define KV_CACHE_MIN_EFFECTIVE_HITS 0.01
/* A continued checkpoint that is a strict prefix of the incoming store is a
 * routine waypoint on the same path. Keep recent hits meaningful, but make
 * never-hit or stale waypoints cheap victims while pre-evicting for the new
 * store. */
#define KV_CACHE_CONTINUED_PREFIX_MIN_FACTOR 0.05
#define KV_CACHE_CONTINUED_PREFIX_HIT_FACTOR 0.45
/* Cold/evict/shutdown checkpoints are intentional anchors, not just automatic
 * waypoints in a single growing conversation. Give them a soft prior so they
 * survive comparable continued entries, while still allowing pressure and poor
 * density to evict them. */
#define KV_CACHE_ANCHOR_REASON_SCORE_FACTOR 2.0

#define PREFIX_FILE_HEADER 176u
#define PREFIX_FILE_VERSION 1u

typedef enum {
    PREFIX_RAM_ONLY,
    PREFIX_WRITE_QUEUED,
    PREFIX_WRITING,
    PREFIX_DURABLE,
    PREFIX_FAILED,
} prefix_state;

typedef struct prefix_entry {
    char node_id[41];
    char parent_id[41];
    char prefix_sha[41];
    char *text;
    size_t text_len;
    char *path;
    ds4_prefix_node *node;
    uint64_t compat_id;
    uint64_t token_hash;
    uint64_t node_bytes;
    uint64_t file_bytes;
    uint64_t last_used;
    uint32_t parent_tokens;
    uint32_t total_tokens;
    uint32_t child_count;
    uint32_t pin_count;
    prefix_state state;
} prefix_entry;

struct ds4_prefix_cache {
    ds4_kvstore *owner;
    ds4_engine *engine;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_t writer;
    bool writer_started;
    bool stopping;
    bool writer_busy;
    uint64_t clock;
    uint64_t ram_budget;
    uint64_t ram_bytes;
    uint64_t reserved_ram_bytes;
    uint64_t disk_bytes;
    uint64_t reserved_disk_bytes;
    uint64_t reserved_kvc_bytes;
    pthread_mutex_t *inference_mu;
    char *dir;
    prefix_entry **entry;
    size_t len;
    size_t cap;
};

static void prefix_cache_destroy(ds4_kvstore *kc);

static uint64_t prefix_disk_claimed_bytes(const ds4_kvstore *kc) {
    if (!kc || !kc->prefix) return 0;
    struct ds4_prefix_cache *pc = kc->prefix;
    pthread_mutex_lock(&pc->mu);
    uint64_t claimed = pc->disk_bytes;
    if (UINT64_MAX - claimed < pc->reserved_disk_bytes) {
        claimed = UINT64_MAX;
    } else {
        claimed += pc->reserved_disk_bytes;
    }
    if (UINT64_MAX - claimed < pc->reserved_kvc_bytes) {
        claimed = UINT64_MAX;
    } else {
        claimed += pc->reserved_kvc_bytes;
    }
    pthread_mutex_unlock(&pc->mu);
    return claimed;
}

static bool prefix_reserve_kvc_bytes(ds4_kvstore *kc, uint64_t kvc_bytes,
                                     uint64_t incoming_bytes) {
    if (!kc || !kc->prefix || kc->budget_bytes == 0) return true;
    struct ds4_prefix_cache *pc = kc->prefix;
    pthread_mutex_lock(&pc->mu);
    uint64_t total = pc->disk_bytes;
    bool ok = UINT64_MAX - total >= pc->reserved_disk_bytes;
    if (ok) total += pc->reserved_disk_bytes;
    if (ok) ok = UINT64_MAX - total >= pc->reserved_kvc_bytes;
    if (ok) total += pc->reserved_kvc_bytes;
    if (ok) ok = UINT64_MAX - total >= kvc_bytes;
    if (ok) total += kvc_bytes;
    if (ok) ok = UINT64_MAX - total >= incoming_bytes;
    if (ok) total += incoming_bytes;
    ok = ok && total <= kc->budget_bytes &&
         UINT64_MAX - pc->reserved_kvc_bytes >= incoming_bytes;
    if (ok) pc->reserved_kvc_bytes += incoming_bytes;
    pthread_mutex_unlock(&pc->mu);
    return ok;
}

static void prefix_release_kvc_bytes(ds4_kvstore *kc, uint64_t bytes) {
    if (!kc || !kc->prefix || bytes == 0) return;
    struct ds4_prefix_cache *pc = kc->prefix;
    pthread_mutex_lock(&pc->mu);
    if (pc->reserved_kvc_bytes >= bytes) pc->reserved_kvc_bytes -= bytes;
    else pc->reserved_kvc_bytes = 0;
    pthread_cond_broadcast(&pc->cv);
    pthread_mutex_unlock(&pc->mu);
}

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} kv_buf;

static void kv_die(const char *msg) {
    fprintf(stderr, "ds4-kvstore: %s\n", msg);
    exit(1);
}

static void *kv_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) kv_die("out of memory");
    return p;
}

static void *kv_xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) kv_die("out of memory");
    return p;
}

static char *kv_xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = kv_xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static void kv_buf_reserve(kv_buf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1) kv_die("buffer overflow");
    size_t need = b->len + add + 1;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < need) cap *= 2;
    b->ptr = kv_xrealloc(b->ptr, cap);
    b->cap = cap;
}

static void kv_buf_append(kv_buf *b, const void *p, size_t n) {
    kv_buf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void kv_buf_putc(kv_buf *b, char c) {
    kv_buf_append(b, &c, 1);
}

static void kv_buf_puts(kv_buf *b, const char *s) {
    kv_buf_append(b, s, strlen(s));
}

static void kv_buf_printf(kv_buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) kv_die("vsnprintf failed");
    kv_buf_reserve(b, (size_t)n);
    vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}

static char *kv_buf_take(kv_buf *b) {
    if (!b->ptr) return kv_xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static double kv_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static const char *kv_log_name(const ds4_kvstore *kc) {
    return kc && kc->log_name ? kc->log_name : "ds4";
}

static void kv_logf(ds4_kvstore *kc, ds4_kvstore_log_type type,
                    const char *fmt, ...) {
    if (!kc || !kc->log) return;
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) {
        va_end(ap2);
        return;
    }
    char *msg = kv_xmalloc((size_t)n + 1);
    vsnprintf(msg, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    kc->log(kc->log_ud, type, msg);
    free(msg);
}

ds4_kvstore_options ds4_kvstore_default_options(void) {
    return (ds4_kvstore_options){
        .min_tokens = KV_CACHE_DEFAULT_MIN_TOKENS,
        .cold_max_tokens = KV_CACHE_DEFAULT_COLD_MAX_TOKENS,
        .continued_interval_tokens = KV_CACHE_DEFAULT_CONTINUED_INTERVAL_TOKENS,
        .boundary_trim_tokens = KV_CACHE_DEFAULT_BOUNDARY_TRIM_TOKENS,
        .boundary_align_tokens = KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS,
    };
}

uint8_t ds4_kvstore_reason_code(const char *reason) {
    if (!reason) return DS4_KVSTORE_REASON_UNKNOWN;
    if (!strcmp(reason, "cold")) return DS4_KVSTORE_REASON_COLD;
    if (!strcmp(reason, "continued")) return DS4_KVSTORE_REASON_CONTINUED;
    if (!strcmp(reason, "evict")) return DS4_KVSTORE_REASON_EVICT;
    if (!strcmp(reason, "shutdown")) return DS4_KVSTORE_REASON_SHUTDOWN;
    if (!strcmp(reason, "agent-system")) return DS4_KVSTORE_REASON_AGENT_SYSTEM;
    if (!strcmp(reason, "agent-session")) return DS4_KVSTORE_REASON_AGENT_SESSION;
    return DS4_KVSTORE_REASON_UNKNOWN;
}

const char *ds4_kvstore_key_kind(uint8_t ext_flags) {
    if (ext_flags & DS4_KVSTORE_EXT_RESPONSES_VISIBLE) return "responses-visible";
    if (ext_flags & DS4_KVSTORE_EXT_THINKING_VISIBLE) return "thinking-visible";
    return "token-text";
}

void ds4_kvstore_le_put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static void kv_le_put64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

uint32_t ds4_kvstore_le_get32(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t kv_le_get64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

typedef struct {
    uint32_t h[5];
    uint64_t bytes;
    uint8_t block[64];
    size_t used;
} sha1_ctx;

static uint32_t rol32(uint32_t v, int n) {
    return (v << n) | (v >> (32 - n));
}

static void sha1_transform(sha1_ctx *c, const uint8_t block[64]) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 80; i++)
        w[i] = rol32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

    uint32_t a = c->h[0], b = c->h[1], d = c->h[3], e = c->h[4];
    uint32_t cc = c->h[2];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) {
            f = (b & cc) | ((~b) & d);
            k = 0x5a827999u;
        } else if (i < 40) {
            f = b ^ cc ^ d;
            k = 0x6ed9eba1u;
        } else if (i < 60) {
            f = (b & cc) | (b & d) | (cc & d);
            k = 0x8f1bbcdcu;
        } else {
            f = b ^ cc ^ d;
            k = 0xca62c1d6u;
        }
        uint32_t tmp = rol32(a, 5) + f + e + k + w[i];
        e = d;
        d = cc;
        cc = rol32(b, 30);
        b = a;
        a = tmp;
    }
    c->h[0] += a;
    c->h[1] += b;
    c->h[2] += cc;
    c->h[3] += d;
    c->h[4] += e;
}

static void sha1_init(sha1_ctx *c) {
    c->h[0] = 0x67452301u;
    c->h[1] = 0xefcdab89u;
    c->h[2] = 0x98badcfeu;
    c->h[3] = 0x10325476u;
    c->h[4] = 0xc3d2e1f0u;
    c->bytes = 0;
    c->used = 0;
}

static void sha1_update(sha1_ctx *c, const void *ptr, size_t len) {
    const uint8_t *p = ptr;
    c->bytes += len;
    while (len != 0) {
        size_t n = 64 - c->used;
        if (n > len) n = len;
        memcpy(c->block + c->used, p, n);
        c->used += n;
        p += n;
        len -= n;
        if (c->used == 64) {
            sha1_transform(c, c->block);
            c->used = 0;
        }
    }
}

static void sha1_final(sha1_ctx *c, uint8_t out[20]) {
    uint64_t bits = c->bytes * 8;
    uint8_t one = 0x80;
    uint8_t zero = 0;
    sha1_update(c, &one, 1);
    while (c->used != 56) sha1_update(c, &zero, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; i++) len[7 - i] = (uint8_t)(bits >> (8 * i));
    sha1_update(c, len, sizeof(len));
    for (int i = 0; i < 5; i++) {
        out[i * 4] = (uint8_t)(c->h[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(c->h[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(c->h[i] >> 8);
        out[i * 4 + 3] = (uint8_t)c->h[i];
    }
}

static void hex20(const uint8_t in[20], char out[41]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 20; i++) {
        out[i * 2] = hex[in[i] >> 4];
        out[i * 2 + 1] = hex[in[i] & 15];
    }
    out[40] = '\0';
}

void ds4_kvstore_sha1_bytes_hex(const void *ptr, size_t len, char out[41]) {
    sha1_ctx c;
    sha1_init(&c);
    sha1_update(&c, ptr, len);
    uint8_t digest[20];
    sha1_final(&c, digest);
    hex20(digest, out);
}

bool ds4_kvstore_sha_hex_name(const char *name, char sha[41]) {
    if (strlen(name) != 43 || strcmp(name + 40, ".kv")) return false;
    for (int i = 0; i < 40; i++) {
        if (!isxdigit((unsigned char)name[i])) return false;
        sha[i] = (char)tolower((unsigned char)name[i]);
    }
    sha[40] = '\0';
    return true;
}

char *ds4_kvstore_path_join(const char *dir, const char *name) {
    kv_buf b = {0};
    kv_buf_puts(&b, dir);
    if (b.len == 0 || b.ptr[b.len - 1] != '/') kv_buf_putc(&b, '/');
    kv_buf_puts(&b, name);
    return kv_buf_take(&b);
}

char *ds4_kvstore_path_for_sha(ds4_kvstore *kc, const char sha[41]) {
    char name[44];
    memcpy(name, sha, 40);
    memcpy(name + 40, ".kv", 4);
    return ds4_kvstore_path_join(kc->dir, name);
}

static bool kv_mkdir_p(const char *path) {
    if (!path || !path[0]) return false;
    char *tmp = kv_xstrdup(path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0700) != 0 && errno != EEXIST) {
            free(tmp);
            return false;
        }
        *p = '/';
    }
    bool ok = mkdir(tmp, 0700) == 0 || errno == EEXIST;
    free(tmp);
    return ok;
}

void ds4_kvstore_entry_free(ds4_kvstore_entry *e) {
    free(e->path);
    memset(e, 0, sizeof(*e));
}

void ds4_kvstore_clear(ds4_kvstore *kc) {
    for (int i = 0; i < kc->len; i++) ds4_kvstore_entry_free(&kc->entry[i]);
    free(kc->entry);
    kc->entry = NULL;
    kc->len = 0;
    kc->cap = 0;
}

static void kv_cache_push(ds4_kvstore *kc, ds4_kvstore_entry e) {
    if (kc->len == kc->cap) {
        kc->cap = kc->cap ? kc->cap * 2 : 16;
        kc->entry = kv_xrealloc(kc->entry, (size_t)kc->cap * sizeof(kc->entry[0]));
    }
    kc->entry[kc->len++] = e;
}

void ds4_kvstore_fill_header(uint8_t h[DS4_KVSTORE_FIXED_HEADER],
                             uint8_t model_id, uint8_t quant_bits,
                             uint8_t reason, uint8_t ext_flags,
                             uint32_t tokens, uint32_t hits, uint32_t ctx_size,
                             uint64_t created_at, uint64_t last_used,
                             uint64_t payload_bytes) {
    memset(h, 0, DS4_KVSTORE_FIXED_HEADER);
    h[0] = KV_CACHE_MAGIC0;
    h[1] = KV_CACHE_MAGIC1;
    h[2] = KV_CACHE_MAGIC2;
    h[3] = KV_CACHE_VERSION;
    h[4] = quant_bits;
    h[5] = reason;
    h[6] = ext_flags;
    h[7] = model_id;
    ds4_kvstore_le_put32(h + 8, tokens);
    ds4_kvstore_le_put32(h + 12, hits);
    ds4_kvstore_le_put32(h + 16, ctx_size);
    h[20] = KV_CACHE_PAYLOAD_ABI;
    kv_le_put64(h + 24, created_at);
    kv_le_put64(h + 32, last_used);
    kv_le_put64(h + 40, payload_bytes);
}

bool ds4_kvstore_read_header(FILE *fp, ds4_kvstore_entry *e,
                             uint32_t *text_bytes) {
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    if (fread(h, 1, sizeof(h), fp) != sizeof(h)) return false;
    if (h[0] != KV_CACHE_MAGIC0 || h[1] != KV_CACHE_MAGIC1 ||
        h[2] != KV_CACHE_MAGIC2 || h[3] != KV_CACHE_VERSION) return false;
    if (h[20] != KV_CACHE_PAYLOAD_ABI) return false;
    e->quant_bits = h[4];
    e->reason = h[5] <= DS4_KVSTORE_REASON_AGENT_SESSION ? h[5] :
                DS4_KVSTORE_REASON_UNKNOWN;
    e->ext_flags = h[6];
    e->model_id = h[7];
    e->tokens = ds4_kvstore_le_get32(h + 8);
    e->hits = ds4_kvstore_le_get32(h + 12);
    e->ctx_size = ds4_kvstore_le_get32(h + 16);
    e->created_at = kv_le_get64(h + 24);
    e->last_used = kv_le_get64(h + 32);
    e->payload_bytes = kv_le_get64(h + 40);
    uint8_t tb[4];
    if (fread(tb, 1, sizeof(tb), fp) != sizeof(tb)) return false;
    *text_bytes = ds4_kvstore_le_get32(tb);
    e->text_bytes = *text_bytes;
    return e->tokens != 0 && (e->quant_bits == 2 || e->quant_bits == 4);
}

bool ds4_kvstore_read_entry_file(const char *path, const char sha[41],
                                 ds4_kvstore_entry *out) {
    struct stat st;
    if (stat(path, &st) != 0 ||
        st.st_size < (off_t)(DS4_KVSTORE_FIXED_HEADER + 4))
        return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    ds4_kvstore_entry e = {0};
    uint32_t text_bytes = 0;
    bool ok = ds4_kvstore_read_header(fp, &e, &text_bytes);
    fclose(fp);
    if (!ok) return false;
    const uint64_t fixed = DS4_KVSTORE_FIXED_HEADER + 4ull;
    if (UINT64_MAX - fixed < (uint64_t)text_bytes ||
        UINT64_MAX - fixed - (uint64_t)text_bytes < e.payload_bytes)
        return false;
    const uint64_t expected = fixed + (uint64_t)text_bytes + e.payload_bytes;
    if ((uint64_t)st.st_size < expected) return false;
    memcpy(e.sha, sha, 41);
    e.path = kv_xstrdup(path);
    e.file_size = (uint64_t)st.st_size;
    *out = e;
    return true;
}

static void kv_cache_refresh(ds4_kvstore *kc) {
    if (!kc->enabled) return;
    ds4_kvstore_clear(kc);
    DIR *d = opendir(kc->dir);
    if (!d) return;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        char *path = ds4_kvstore_path_join(kc->dir, de->d_name);
        ds4_kvstore_entry e = {0};
        if (ds4_kvstore_read_entry_file(path, sha, &e)) kv_cache_push(kc, e);
        free(path);
    }
    closedir(d);
}

bool ds4_kvstore_touch_file(const char *path, uint32_t hits) {
    FILE *fp = fopen(path, "r+b");
    if (!fp) return false;
    ds4_kvstore_entry e = {0};
    uint32_t text_bytes = 0;
    bool ok = ds4_kvstore_read_header(fp, &e, &text_bytes);
    if (ok) {
        uint8_t h[DS4_KVSTORE_FIXED_HEADER];
        uint64_t now = (uint64_t)time(NULL);
        ds4_kvstore_fill_header(h, e.model_id, e.quant_bits, e.reason, e.ext_flags,
                                e.tokens, hits, e.ctx_size,
                                e.created_at, now, e.payload_bytes);
        ok = fseek(fp, 0, SEEK_SET) == 0 &&
             fwrite(h, 1, sizeof(h), fp) == sizeof(h);
    }
    fclose(fp);
    return ok;
}

static bool kv_cache_incoming_supersedes_continued(
        const ds4_kvstore_entry *e,
        const ds4_kvstore_eviction_context *incoming) {
    if (!e || !incoming || !incoming->text) return false;
    if (e->reason != DS4_KVSTORE_REASON_CONTINUED) return false;
    if (e->text_bytes == 0 || e->text_bytes > SIZE_MAX) return false;
    if ((size_t)e->text_bytes >= incoming->text_len) return false;
    if (e->model_id != incoming->model_id) return false;
    if (incoming->reject_different_quant &&
        e->quant_bits != incoming->quant_bits)
        return false;
    /* A smaller-context checkpoint can be loaded by a larger context, but not
     * the reverse.  The incoming checkpoint dominates this one only if it is at
     * least as widely reusable. */
    if (incoming->ctx_size > e->ctx_size) return false;

    char prefix_sha[41];
    ds4_kvstore_sha1_bytes_hex(incoming->text, (size_t)e->text_bytes,
                               prefix_sha);
    return !strcmp(prefix_sha, e->sha);
}

static bool kv_cache_reason_is_anchor(uint8_t reason) {
    return reason == DS4_KVSTORE_REASON_COLD ||
           reason == DS4_KVSTORE_REASON_EVICT ||
           reason == DS4_KVSTORE_REASON_SHUTDOWN;
}

double ds4_kvstore_entry_eviction_score(
        const ds4_kvstore_entry *e,
        const ds4_tokens *live,
        uint64_t now,
        const ds4_kvstore_eviction_context *incoming) {
    if (!e || e->file_size == 0) return 0.0;
    (void)live;
    double effective_hits = (double)e->hits;
    uint64_t used_at = e->last_used ? e->last_used : e->created_at;
    if (used_at == 0) {
        effective_hits = 0.0;
    } else if (now > used_at) {
        double elapsed = (double)(now - used_at);
        effective_hits *= exp2(-elapsed / (double)DS4_KVSTORE_HIT_HALF_LIFE_SECONDS);
        if (effective_hits < KV_CACHE_MIN_EFFECTIVE_HITS) effective_hits = 0.0;
    }
    double score = (effective_hits + 1.0) *
                   (double)e->tokens / (double)e->file_size;
    if (kv_cache_reason_is_anchor(e->reason))
        score *= KV_CACHE_ANCHOR_REASON_SCORE_FACTOR;
    if (kv_cache_incoming_supersedes_continued(e, incoming)) {
        double h = effective_hits > 0.0 ?
            effective_hits / (effective_hits + 1.0) : 0.0;
        score *= KV_CACHE_CONTINUED_PREFIX_MIN_FACTOR +
                 KV_CACHE_CONTINUED_PREFIX_HIT_FACTOR * h;
    }
    return score;
}

void ds4_kvstore_evict(ds4_kvstore *kc, const ds4_tokens *live,
                       uint64_t extra_bytes,
                       const ds4_kvstore_eviction_context *incoming) {
    if (!kc->enabled || kc->budget_bytes == 0) return;
    if (extra_bytes > kc->budget_bytes) return;
    kv_cache_refresh(kc);
    const uint64_t now = (uint64_t)time(NULL);
    uint64_t total = 0;
    for (int i = 0; i < kc->len; i++) total += kc->entry[i].file_size;
    const uint64_t tree_claimed = prefix_disk_claimed_bytes(kc);
    uint64_t target = kc->budget_bytes - extra_bytes;
    if (tree_claimed >= target) target = 0;
    else target -= tree_claimed;
    while (total > target && kc->len > 0) {
        int victim = 0;
        double victim_score =
            ds4_kvstore_entry_eviction_score(&kc->entry[0], live, now,
                                             incoming);
        for (int i = 1; i < kc->len; i++) {
            double score =
                ds4_kvstore_entry_eviction_score(&kc->entry[i], live, now,
                                                 incoming);
            if (score < victim_score ||
                (score == victim_score &&
                 kc->entry[i].last_used < kc->entry[victim].last_used))
            {
                victim = i;
                victim_score = score;
            }
        }
        ds4_kvstore_entry e = kc->entry[victim];
        if (unlink(e.path) == 0) {
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache evicted reason=disk-cache-full tokens=%u hits=%u size=%.2f MiB file=%s",
                    kv_log_name(kc),
                    e.tokens,
                    e.hits,
                    (double)e.file_size / (1024.0 * 1024.0),
                    e.path ? e.path : "?");
            if (total >= e.file_size) total -= e.file_size;
            else total = 0;
        } else {
            total = 0;
        }
        ds4_kvstore_entry_free(&e);
        memmove(kc->entry + victim, kc->entry + victim + 1,
                (size_t)(kc->len - victim - 1) * sizeof(kc->entry[0]));
        kc->len--;
    }
}

bool ds4_kvstore_open(ds4_kvstore *kc, const char *dir, uint64_t budget_mb,
                      bool reject_different_quant, ds4_kvstore_options opt,
                      const char *log_name,
                      void (*log)(void *ud, ds4_kvstore_log_type type, const char *msg),
                      void *log_ud) {
    memset(kc, 0, sizeof(*kc));
    if (!dir) return false;
    kc->log_name = log_name;
    kc->log = log;
    kc->log_ud = log_ud;
    if (!kv_mkdir_p(dir)) {
        kv_logf(kc, DS4_KVSTORE_LOG_DEFAULT,
                "%s: failed to create KV cache directory %s: %s",
                kv_log_name(kc), dir, strerror(errno));
        return false;
    }
    kc->enabled = true;
    kc->dir = kv_xstrdup(dir);
    if (budget_mb == 0) budget_mb = DS4_KVSTORE_DEFAULT_MB;
    kc->budget_bytes = budget_mb * 1024ull * 1024ull;
    kc->reject_different_quant = reject_different_quant;
    kc->opt = opt;
    ds4_kvstore_evict(kc, NULL, 0, NULL);
    kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
            "%s: KV disk cache %s (budget=%llu MiB, cross-quant=%s, min=%d, cold_max=%d, continued=%d, trim=%d, align=%d, hit_half_life=%llus)",
            kv_log_name(kc),
            kc->dir,
            (unsigned long long)(kc->budget_bytes / (1024ull * 1024ull)),
            reject_different_quant ? "reject" : "accept",
            kc->opt.min_tokens,
            kc->opt.cold_max_tokens,
            kc->opt.continued_interval_tokens,
            kc->opt.boundary_trim_tokens,
            kc->opt.boundary_align_tokens,
            (unsigned long long)DS4_KVSTORE_HIT_HALF_LIFE_SECONDS);
    return true;
}

void ds4_kvstore_close(ds4_kvstore *kc) {
    prefix_cache_destroy(kc);
    ds4_kvstore_clear(kc);
    free(kc->dir);
    memset(kc, 0, sizeof(*kc));
}

char *ds4_kvstore_render_tokens_text(ds4_engine *engine,
                                     const ds4_tokens *tokens,
                                     size_t *out_len) {
    kv_buf b = {0};
    for (int i = 0; i < tokens->len; i++) {
        size_t len = 0;
        char *piece = ds4_token_text(engine, tokens->v[i], &len);
        kv_buf_append(&b, piece, len);
        free(piece);
    }
    if (out_len) *out_len = b.len;
    return kv_buf_take(&b);
}

bool ds4_kvstore_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len) {
    return prefix_len <= text_len &&
           (prefix_len == 0 || memcmp(text, prefix, prefix_len) == 0);
}

void ds4_kvstore_tokens_copy_prefix(ds4_tokens *dst, const ds4_tokens *src, int n) {
    dst->len = 0;
    if (!src) return;
    if (n > src->len) n = src->len;
    for (int i = 0; i < n; i++) ds4_tokens_push(dst, src->v[i]);
}

static void tokens_append(ds4_tokens *dst, const ds4_tokens *src) {
    if (!dst || !src) return;
    for (int i = 0; i < src->len; i++) ds4_tokens_push(dst, src->v[i]);
}

void ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
        ds4_engine *engine,
        const ds4_tokens *exact_prefix,
        const char *suffix_text,
        ds4_tokens *out) {
    ds4_tokens_copy(out, exact_prefix);

    ds4_tokens suffix = {0};
    /* The suffix may start with DS4 chat markers such as <｜User｜> or
     * </think>, so use the rendered-chat tokenizer, not plain text BPE. */
    ds4_tokenize_rendered_chat(engine, suffix_text ? suffix_text : "", &suffix);
    tokens_append(out, &suffix);
    ds4_tokens_free(&suffix);
}

int ds4_kvstore_store_len(const ds4_kvstore *kc, int tokens) {
    const int trim = kc->opt.boundary_trim_tokens;
    const int align = kc->opt.boundary_align_tokens;
    if (tokens > kc->opt.min_tokens + trim) {
        int stable = tokens - trim;
        if (align > 0) stable -= stable % align;
        if (stable >= kc->opt.min_tokens) return stable;
    }
    return tokens;
}

int ds4_kvstore_chat_anchor_pos(const ds4_kvstore *kc,
                                const ds4_tokens *prompt,
                                int user_token_id,
                                int assistant_token_id) {
    if (!prompt || user_token_id < 0 || assistant_token_id < 0) return -1;

    /* Cold checkpoints maximize reuse across independent agent sessions.  The
     * stable rendered chat prefix is everything before the user message that
     * asks this specific task.  Some clients put stable user-role scaffolding
     * first, so use the last user marker before the first assistant marker. */
    int last_user = -1;
    for (int i = 0; i < prompt->len; i++) {
        const int token = prompt->v[i];
        if (token == assistant_token_id) break;
        if (token == user_token_id) last_user = i;
    }
    return last_user >= kc->opt.min_tokens ? last_user : -1;
}

static int kv_cache_continued_step(const ds4_kvstore *kc) {
    if (!kc->enabled || kc->opt.continued_interval_tokens <= 0) return 0;
    int step = kc->opt.continued_interval_tokens;
    const int align = kc->opt.boundary_align_tokens;
    if (align > 0) {
        step = ((step + align - 1) / align) * align;
        if (step <= 0) step = align;
    }
    return step;
}

int ds4_kvstore_continued_store_target(const ds4_kvstore *kc, int live_tokens) {
    const int step = kv_cache_continued_step(kc);
    if (step <= 0) return 0;
    if (live_tokens < kc->opt.min_tokens) return 0;
    if (live_tokens % step != 0) return 0;
    if (live_tokens <= kc->continued_last_store_tokens) return 0;
    return live_tokens;
}

void ds4_kvstore_note_store(ds4_kvstore *kc, int tokens) {
    if (tokens > kc->continued_last_store_tokens) {
        kc->continued_last_store_tokens = tokens;
    }
}

int ds4_kvstore_suppress_continued_store(ds4_kvstore *kc, int tokens) {
    if (ds4_kvstore_continued_store_target(kc, tokens) != tokens) return -1;
    int old = kc->continued_last_store_tokens;
    ds4_kvstore_note_store(kc, tokens);
    return old;
}

void ds4_kvstore_restore_suppressed_continued(ds4_kvstore *kc,
                                              int old_tokens,
                                              int suppressed_tokens) {
    if (old_tokens >= 0 && kc->continued_last_store_tokens == suppressed_tokens) {
        kc->continued_last_store_tokens = old_tokens;
    }
}

static bool kv_cache_file_size_bytes(uint64_t text_bytes,
                                     uint64_t payload_bytes,
                                     uint64_t trailer_bytes,
                                     uint64_t *file_bytes) {
    const uint64_t fixed = DS4_KVSTORE_FIXED_HEADER + 4ull;
    if (UINT64_MAX - fixed < text_bytes ||
        UINT64_MAX - fixed - text_bytes < payload_bytes ||
        UINT64_MAX - fixed - text_bytes - payload_bytes < trailer_bytes)
        return false;
    if (file_bytes) *file_bytes = fixed + text_bytes + payload_bytes + trailer_bytes;
    return true;
}

static bool kv_cache_budget_required(uint64_t file_bytes,
                                     uint64_t *required_bytes) {
    /* The serialized size is deterministic for one snapshot, including the
     * optional trailer.  Reserve 1% headroom so filesystem/accounting surprises
     * cannot produce a file that is immediately removed by the budget pass. */
    uint64_t slack = file_bytes / 100u;
    if (file_bytes % 100u) slack++;
    if (UINT64_MAX - file_bytes < slack) return false;
    if (required_bytes) *required_bytes = file_bytes + slack;
    return true;
}

bool ds4_kvstore_file_size_fits(const ds4_kvstore *kc,
                                uint64_t text_bytes,
                                uint64_t payload_bytes,
                                uint64_t trailer_bytes,
                                uint64_t *file_bytes_out,
                                uint64_t *required_bytes_out) {
    uint64_t file_bytes = 0;
    if (!kv_cache_file_size_bytes(text_bytes, payload_bytes, trailer_bytes,
                                  &file_bytes))
        return false;
    if (file_bytes_out) *file_bytes_out = file_bytes;
    if (!kc || kc->budget_bytes == 0) return true;
    uint64_t required = 0;
    if (!kv_cache_budget_required(file_bytes, &required)) return false;
    if (required_bytes_out) *required_bytes_out = required;
    return required <= kc->budget_bytes;
}

static bool kv_cache_file_text_matches(const char *path, const char sha[41],
                                       const char *text, size_t text_len) {
    if (text_len > UINT32_MAX) return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;

    ds4_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes) &&
              text_bytes == (uint32_t)text_len;
    char *stored = NULL;
    if (ok) {
        stored = kv_xmalloc((size_t)text_bytes + 1);
        ok = fread(stored, 1, text_bytes, fp) == text_bytes;
    }
    fclose(fp);
    if (!ok) {
        free(stored);
        return false;
    }

    char stored_sha[41];
    ds4_kvstore_sha1_bytes_hex(stored, text_bytes, stored_sha);
    ok = !strcmp(stored_sha, sha) &&
         (text_len == 0 || memcmp(stored, text, text_len) == 0);
    free(stored);
    return ok;
}

static bool kv_cache_existing_compatible(ds4_kvstore *kc, const char *path,
                                         const char sha[41],
                                         const char *text, size_t text_len,
                                         int model_id, int quant_bits, int ctx_size) {
    if (access(path, F_OK) != 0) return false;
    ds4_kvstore_entry e = {0};
    if (!ds4_kvstore_read_entry_file(path, sha, &e)) return false;
    bool compatible = e.model_id == (uint8_t)model_id &&
                      (!kc->reject_different_quant ||
                       e.quant_bits == (uint8_t)quant_bits) &&
                      e.ctx_size <= (uint32_t)ctx_size &&
                      kv_cache_file_text_matches(path, sha, text, text_len);
    ds4_kvstore_entry_free(&e);
    if (!compatible) {
        if (unlink(path) == 0) {
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache replaced incompatible file %s",
                    kv_log_name(kc), path);
        }
        return false;
    }
    return true;
}

static bool kv_trailer_serialized_size(const ds4_kvstore_trailer_hooks *hooks,
                                       const char *text,
                                       uint64_t *bytes_out) {
    if (bytes_out) *bytes_out = 0;
    if (!hooks || !hooks->serialized_size) return true;
    return hooks->serialized_size(hooks->ud, text, bytes_out);
}

static bool kv_trailer_write(const ds4_kvstore_trailer_hooks *hooks,
                             FILE *fp, const char *text,
                             uint64_t *written_bytes) {
    if (written_bytes) *written_bytes = 0;
    if (!hooks || !hooks->write) return true;
    return hooks->write(hooks->ud, fp, text, written_bytes);
}

static void kv_cache_rewrite_trailer(ds4_kvstore *kc, const char *path,
                                     const char *text,
                                     const ds4_kvstore_trailer_hooks *hooks) {
    uint64_t trailer_est = 0;
    if (!hooks || !hooks->write || !hooks->serialized_size ||
        !kv_trailer_serialized_size(hooks, text, &trailer_est) ||
        trailer_est == 0)
    {
        return;
    }
    FILE *fp = fopen(path, "r+b");
    if (!fp) return;
    ds4_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes);
    uint64_t end = DS4_KVSTORE_FIXED_HEADER + 4ull +
                   (uint64_t)text_bytes + hdr.payload_bytes;
    if (ok && end <= (uint64_t)INT64_MAX &&
        fseeko(fp, (off_t)end, SEEK_SET) == 0 &&
        ftruncate(fileno(fp), (off_t)end) == 0)
    {
        uint64_t ignored = 0;
        ok = kv_trailer_write(hooks, fp, text, &ignored) && fflush(fp) == 0;
        if (ok && ignored > 0) {
            uint8_t h[DS4_KVSTORE_FIXED_HEADER];
            uint64_t now = (uint64_t)time(NULL);
            ds4_kvstore_fill_header(h, hdr.model_id, hdr.quant_bits, hdr.reason,
                                    (uint8_t)(hdr.ext_flags | hooks->ext_flag),
                                    hdr.tokens, hdr.hits, hdr.ctx_size,
                                    hdr.created_at, now, hdr.payload_bytes);
            ok = fseeko(fp, 0, SEEK_SET) == 0 &&
                 fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
                 fflush(fp) == 0;
        }
    }
    fclose(fp);
    (void)kc;
    (void)ok;
}

bool ds4_kvstore_store_live_prefix_text(ds4_kvstore *kc,
                                        ds4_engine *engine,
                                        ds4_session *session,
                                        const ds4_tokens *tokens,
                                        int store_len,
                                        const char *reason,
                                        const char *cache_text_override,
                                        uint8_t cache_text_ext,
                                        const char *cache_text_key,
                                        const ds4_kvstore_trailer_hooks *hooks,
                                        char *err,
                                        size_t err_len) {
    if (!kc->enabled) return false;
    if (!tokens || store_len < kc->opt.min_tokens) return false;
    const int original_len = tokens->len;

    ds4_tokens store_tokens = {0};
    ds4_kvstore_tokens_copy_prefix(&store_tokens, tokens, store_len);

    const int quant_bits = ds4_engine_routed_quant_bits(engine);
    if (quant_bits != 2 && quant_bits != 4) {
        ds4_tokens_free(&store_tokens);
        return false;
    }
    const int model_id = ds4_engine_model_id(engine);

    char save_err[160] = {0};
    const ds4_tokens *live_tokens = ds4_session_tokens(session);
    if (!live_tokens ||
        live_tokens->len != store_tokens.len ||
        !ds4_tokens_starts_with(live_tokens, &store_tokens))
    {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because live checkpoint is at %d",
                kv_log_name(kc),
                store_tokens.len,
                reason,
                live_tokens ? live_tokens->len : -1);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    size_t text_len = 0;
    char *text = NULL;
    const bool text_override = cache_text_override && cache_text_override[0];
    if (text_override) {
        text = kv_xstrdup(cache_text_override);
        text_len = strlen(text);
    } else {
        text = ds4_kvstore_render_tokens_text(engine, &store_tokens, &text_len);
    }
    if (text_len > UINT32_MAX) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d because rendered text is too large",
                kv_log_name(kc), store_tokens.len);
        free(text);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    uint64_t trailer_est_bytes = 0;
    if (!kv_trailer_serialized_size(hooks, text, &trailer_est_bytes)) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because tool map size overflowed",
                kv_log_name(kc), store_tokens.len, reason);
        free(text);
        ds4_tokens_free(&store_tokens);
        return false;
    }
    char sha[41];
    ds4_kvstore_sha1_bytes_hex(text, text_len, sha);
    char *path = ds4_kvstore_path_for_sha(kc, sha);
    const uint8_t reason_code = ds4_kvstore_reason_code(reason);

    if (kv_cache_existing_compatible(kc, path, sha, text, text_len,
                                     model_id,
                                     quant_bits, ds4_session_ctx(session))) {
        kv_cache_rewrite_trailer(kc, path, text, hooks);
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return true;
    }

    ds4_session_payload_file staged = {0};
    if (ds4_session_stage_payload(session, &staged,
                                  save_err, sizeof(save_err)) != 0) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because KV payload staging failed: %s",
                kv_log_name(kc),
                store_tokens.len,
                reason,
                save_err[0] ? save_err : "unknown error");
        if (err && err_len) snprintf(err, err_len, "%s",
                                     save_err[0] ? save_err : "unknown error");
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return false;
    }
    uint64_t payload_bytes = staged.bytes;

    uint64_t est_file_bytes = 0, est_required_bytes = 0;
    if (!ds4_kvstore_file_size_fits(kc, (uint64_t)text_len, payload_bytes,
                                    trailer_est_bytes,
                                    &est_file_bytes, &est_required_bytes)) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because estimated file size %.2f MiB (%.2f MiB with safety) exceeds budget %.2f MiB",
                kv_log_name(kc),
                store_tokens.len,
                reason,
                (double)est_file_bytes / (1024.0 * 1024.0),
                (double)est_required_bytes / (1024.0 * 1024.0),
                (double)kc->budget_bytes / (1024.0 * 1024.0));
        ds4_session_payload_file_free(&staged);
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    ds4_kvstore_eviction_context incoming = {
        .text = text,
        .text_len = text_len,
        .model_id = (uint8_t)model_id,
        .quant_bits = (uint8_t)quant_bits,
        .ctx_size = (uint32_t)ds4_session_ctx(session),
        .reject_different_quant = kc->reject_different_quant,
    };
    ds4_kvstore_evict(kc, live_tokens, est_file_bytes, &incoming);

    uint64_t kvc_bytes = 0;
    for (int i = 0; i < kc->len; i++) {
        if (UINT64_MAX - kvc_bytes < kc->entry[i].file_size) {
            kvc_bytes = UINT64_MAX;
            break;
        }
        kvc_bytes += kc->entry[i].file_size;
    }
    if (!prefix_reserve_kvc_bytes(kc, kvc_bytes, est_file_bytes)) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because the shared KVC/tree disk budget is full",
                kv_log_name(kc), store_tokens.len, reason);
        ds4_session_payload_file_free(&staged);
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    kv_buf tmpb = {0};
    kv_buf_printf(&tmpb, "%s.tmp.%ld", path, (long)getpid());
    char *tmp = kv_buf_take(&tmpb);
    const double save_t0 = kv_now_sec();
    FILE *fp = fopen(tmp, "wb");
    if (!fp) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache failed to create %s: %s save=%.1f ms",
                kv_log_name(kc), tmp, strerror(errno),
                (kv_now_sec() - save_t0) * 1000.0);
        ds4_session_payload_file_free(&staged);
        prefix_release_kvc_bytes(kc, est_file_bytes);
        free(tmp);
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    const uint64_t now = (uint64_t)time(NULL);
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    uint8_t ext_flags = trailer_est_bytes > 0 && hooks ? hooks->ext_flag : 0;
    if (text_override) ext_flags |= cache_text_ext;
    ds4_kvstore_fill_header(h, (uint8_t)model_id, (uint8_t)quant_bits,
                            reason_code, ext_flags,
                            (uint32_t)store_tokens.len, 0,
                            (uint32_t)ds4_session_ctx(session),
                            now, now, payload_bytes);
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, (uint32_t)text_len);
    uint64_t trailer_bytes = 0;
    errno = 0;
    bool ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
              fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
              fwrite(text, 1, text_len, fp) == text_len &&
              ds4_session_write_staged_payload(&staged, fp,
                                               save_err, sizeof(save_err)) == 0 &&
              kv_trailer_write(hooks, fp, text, &trailer_bytes) &&
              fflush(fp) == 0;
    int saved_errno = errno;
    if (fclose(fp) != 0) {
        if (!saved_errno) saved_errno = errno;
        ok = false;
    }
    uint64_t final_file_bytes = 0, final_required_bytes = 0;
    bool final_size_over_budget = false;
    if (ok && !ds4_kvstore_file_size_fits(kc, (uint64_t)text_len, payload_bytes,
                                          trailer_bytes,
                                          &final_file_bytes,
                                          &final_required_bytes))
    {
        final_size_over_budget = true;
        ok = false;
    }
    if (ok && final_file_bytes > est_file_bytes) {
        final_size_over_budget = true;
        ok = false;
    }
    if (ok && rename(tmp, path) != 0) {
        saved_errno = errno;
        ok = false;
    }
    const double save_ms = (kv_now_sec() - save_t0) * 1000.0;
    prefix_release_kvc_bytes(kc, est_file_bytes);
    if (!ok) {
        if (final_size_over_budget) {
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache skipped tokens=%d reason=%s because final file size %.2f MiB (%.2f MiB with safety) exceeds budget %.2f MiB save=%.1f ms",
                    kv_log_name(kc),
                    store_tokens.len,
                    reason,
                    (double)final_file_bytes / (1024.0 * 1024.0),
                    (double)final_required_bytes / (1024.0 * 1024.0),
                    (double)kc->budget_bytes / (1024.0 * 1024.0),
                    save_ms);
        } else {
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache store failed (%s): %s save=%.1f ms",
                    kv_log_name(kc),
                    reason,
                    saved_errno ? strerror(saved_errno) :
                    (save_err[0] ? save_err : "unknown error"),
                    save_ms);
        }
        if (err && err_len) {
            snprintf(err, err_len, "%s",
                     saved_errno ? strerror(saved_errno) :
                     (save_err[0] ? save_err : "unknown error"));
        }
        unlink(tmp);
    } else {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache stored tokens=%d trimmed=%d reason=%s key=%s size=%.2f MiB save=%.1f ms",
                kv_log_name(kc),
                store_tokens.len,
                original_len - store_tokens.len,
                reason,
                text_override ? (cache_text_key ? cache_text_key : "visible-transcript") : "token-text",
                (double)(DS4_KVSTORE_FIXED_HEADER + 4ull + text_len + payload_bytes + trailer_bytes) / (1024.0 * 1024.0),
                save_ms);
    }
    ds4_session_payload_file_free(&staged);
    free(tmp);
    free(text);
    free(path);
    ds4_tokens_free(&store_tokens);
    return ok;
}

bool ds4_kvstore_store_live_prefix(ds4_kvstore *kc,
                                   ds4_engine *engine,
                                   ds4_session *session,
                                   const ds4_tokens *tokens,
                                   int store_len,
                                   const char *reason,
                                   const ds4_kvstore_trailer_hooks *hooks,
                                   char *err,
                                   size_t err_len) {
    return ds4_kvstore_store_live_prefix_text(kc, engine, session, tokens,
                                              store_len, reason, NULL, 0, NULL,
                                              hooks, err, err_len);
}

bool ds4_kvstore_maybe_store_continued(ds4_kvstore *kc,
                                       ds4_engine *engine,
                                       ds4_session *session,
                                       const ds4_kvstore_trailer_hooks *hooks,
                                       char *err,
                                       size_t err_len) {
    const ds4_tokens *tokens = ds4_session_tokens(session);
    if (!tokens) return false;
    const int target = ds4_kvstore_continued_store_target(kc, tokens->len);
    if (target == 0) return false;
    if (ds4_kvstore_store_live_prefix(kc, engine, session, tokens, target,
                                      "continued", hooks, err, err_len))
    {
        ds4_kvstore_note_store(kc, target);
        return true;
    }
    return false;
}

int ds4_kvstore_find_text_prefix(ds4_kvstore *kc, const char *prompt_text,
                                 int model_id, int quant_bits, int ctx_size) {
    if (!prompt_text) return -1;
    const size_t prompt_bytes = strlen(prompt_text);
    kv_cache_refresh(kc);
    int best = -1;
    for (int i = 0; i < kc->len; i++) {
        ds4_kvstore_entry *e = &kc->entry[i];
        if (e->text_bytes > prompt_bytes || e->text_bytes > SIZE_MAX) continue;
        if ((int)e->tokens < kc->opt.min_tokens) continue;
        if (e->model_id != (uint8_t)model_id) continue;
        if ((uint32_t)ctx_size < e->ctx_size) continue;
        if (kc->reject_different_quant && e->quant_bits != (uint8_t)quant_bits) continue;
        if (best >= 0) {
            ds4_kvstore_entry *b = &kc->entry[best];
            if (e->text_bytes < b->text_bytes) continue;
            if (e->text_bytes == b->text_bytes && e->tokens <= b->tokens) continue;
        }
        char sha[41];
        ds4_kvstore_sha1_bytes_hex(prompt_text, (size_t)e->text_bytes, sha);
        if (!strcmp(sha, e->sha)) best = i;
    }
    return best;
}

int ds4_kvstore_try_load_text(ds4_kvstore *kc,
                              ds4_engine *engine,
                              ds4_session *session,
                              const char *prompt_text,
                              ds4_tokens *effective_prompt,
                              ds4_kvstore_load_result *result,
                              const ds4_kvstore_trailer_hooks *hooks,
                              bool responses_protocol) {
    if (result) memset(result, 0, sizeof(*result));
    if (effective_prompt) effective_prompt->len = 0;
    if (!kc->enabled || !prompt_text) return 0;
    const int quant_bits = ds4_engine_routed_quant_bits(engine);
    if (quant_bits != 2 && quant_bits != 4) return 0;
    const int model_id = ds4_engine_model_id(engine);
    const size_t prompt_bytes = strlen(prompt_text);
    int idx = ds4_kvstore_find_text_prefix(kc, prompt_text, model_id, quant_bits,
                                           ds4_session_ctx(session));
    if (idx < 0) return 0;

    ds4_kvstore_entry e = kc->entry[idx];
    char *path = kv_xstrdup(e.path);
    const double load_t0 = kv_now_sec();
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        free(path);
        return 0;
    }
    uint32_t text_bytes = 0;
    ds4_kvstore_entry hdr = {0};
    const char *fail_reason = "invalid header";
    bool header_ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes);
    char *cached_text = NULL;
    if (header_ok) {
        if (hdr.model_id != (uint8_t)model_id) {
            header_ok = false;
            fail_reason = "cached checkpoint was written for a different model";
        } else if ((uint64_t)text_bytes > prompt_bytes) {
            header_ok = false;
            fail_reason = "cached text is longer than prompt";
        } else {
            cached_text = kv_xmalloc((size_t)text_bytes + 1);
            if (fread(cached_text, 1, text_bytes, fp) != text_bytes) {
                header_ok = false;
                fail_reason = "truncated cached text";
            } else {
                cached_text[text_bytes] = '\0';
                char text_sha[41];
                ds4_kvstore_sha1_bytes_hex(cached_text, text_bytes, text_sha);
                if (strcmp(text_sha, e.sha)) {
                    header_ok = false;
                    fail_reason = "cached text hash mismatch";
                } else if (!ds4_kvstore_byte_prefix_match(prompt_text, prompt_bytes,
                                                          cached_text, text_bytes)) {
                    header_ok = false;
                    fail_reason = "cached text prefix mismatch";
                }
            }
        }
    }
    char err[160] = {0};
    int loaded = 0;
    if (header_ok &&
        ds4_session_load_payload(session, fp, hdr.payload_bytes, err, sizeof(err)) == 0)
    {
        const ds4_tokens *loaded_tokens = ds4_session_tokens(session);
        if (loaded_tokens && loaded_tokens->len == (int)hdr.tokens) {
            loaded = (int)hdr.tokens;
            if (effective_prompt) {
                /* The cache lookup was by bytes, but the graph state is still
                 * the exact token history stored in the payload.  Build the
                 * prompt from that exact history and tokenize only the text
                 * suffix after the byte prefix. */
                ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
                    engine, loaded_tokens, prompt_text + text_bytes,
                    effective_prompt);
            }
            if (hooks && hooks->load && (hdr.ext_flags & hooks->ext_flag)) {
                hooks->load(hooks->ud, fp, hooks->load_wanted);
            }
        } else {
            ds4_session_invalidate(session);
            unlink(path);
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache discarded corrupt text-prefix payload%s%s %s",
                    kv_log_name(kc),
                    responses_protocol ? " " : "",
                    responses_protocol ? "RESPPROTO" : "",
                    path);
        }
    } else {
        if (header_ok) ds4_session_invalidate(session);
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache load failed%s%s %s: %s load=%.1f ms",
                kv_log_name(kc),
                responses_protocol ? " " : "",
                responses_protocol ? "RESPPROTO" : "",
                path,
                header_ok ? err : fail_reason,
                (kv_now_sec() - load_t0) * 1000.0);
    }
    fclose(fp);

    if (loaded > 0) {
        const double load_ms = (kv_now_sec() - load_t0) * 1000.0;
        kc->continued_last_store_tokens = loaded;
        const char *key_kind = ds4_kvstore_key_kind(hdr.ext_flags);
        ds4_kvstore_touch_file(path, hdr.hits + 1);
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache hit text%s%s tokens=%d text=%u quant=%u key=%s load=%.1f ms file=%s",
                kv_log_name(kc),
                responses_protocol ? " " : "",
                responses_protocol ? "RESPPROTO" : "",
                loaded, text_bytes, hdr.quant_bits, key_kind, load_ms, path);
        if (result) {
            result->tokens = loaded;
            result->text_bytes = text_bytes;
            result->quant_bits = hdr.quant_bits;
            result->ext_flags = hdr.ext_flags;
            result->load_ms = load_ms;
            result->path = kv_xstrdup(path);
        }
    }
    free(cached_text);
    free(path);
    return loaded;
}

void ds4_kvstore_load_result_free(ds4_kvstore_load_result *result) {
    if (!result) return;
    free(result->path);
    memset(result, 0, sizeof(*result));
}

static void prefix_put64(uint8_t *p, uint64_t v) {
    ds4_kvstore_le_put32(p, (uint32_t)v);
    ds4_kvstore_le_put32(p + 4, (uint32_t)(v >> 32));
}

static uint64_t prefix_get64(const uint8_t *p) {
    return (uint64_t)ds4_kvstore_le_get32(p) |
           ((uint64_t)ds4_kvstore_le_get32(p + 4) << 32);
}

static uint64_t *prefix_prompt_token_hashes(const ds4_tokens *tokens) {
    if (!tokens || tokens->len < 0 ||
        (size_t)tokens->len > SIZE_MAX / sizeof(uint64_t) - 1u)
        return NULL;
    uint64_t *hashes = kv_xmalloc(
        ((size_t)tokens->len + 1u) * sizeof(hashes[0]));
    hashes[0] = UINT64_C(1469598103934665603);
    for (int i = 0; i < tokens->len; i++) {
        const uint32_t token = (uint32_t)tokens->v[i];
        uint64_t h = hashes[i];
        for (unsigned byte = 0; byte < 8; byte++) {
            h ^= (uint8_t)((uint64_t)token >> (byte * 8));
            h *= UINT64_C(1099511628211);
        }
        hashes[i + 1] = h;
    }
    return hashes;
}

static void prefix_node_id(const char *parent_id,
                           const ds4_prefix_node *node,
                           char out[41]) {
    uint8_t identity[72] = {0};
    if (parent_id) memcpy(identity, parent_id, strlen(parent_id));
    uint64_t digest[2] = {0};
    ds4_prefix_node_digest(node, digest);
    prefix_put64(identity + 40, digest[0]);
    prefix_put64(identity + 48, digest[1]);
    prefix_put64(identity + 56, ds4_prefix_node_compat_id(node));
    ds4_kvstore_le_put32(identity + 64,
                         ds4_prefix_node_parent_tokens(node));
    ds4_kvstore_le_put32(identity + 68,
                         ds4_prefix_node_total_tokens(node));
    ds4_kvstore_sha1_bytes_hex(identity, sizeof(identity), out);
}

static prefix_entry *prefix_entry_new(void) {
    prefix_entry *e = kv_xmalloc(sizeof(*e));
    memset(e, 0, sizeof(*e));
    return e;
}

static void prefix_entry_free(prefix_entry *e) {
    if (!e) return;
    free(e->text);
    free(e->path);
    ds4_prefix_node_free(e->node);
    free(e);
}

static prefix_entry *prefix_find_id_locked(struct ds4_prefix_cache *pc,
                                           const char *id) {
    if (!pc || !id || !id[0]) return NULL;
    for (size_t i = 0; i < pc->len; i++) {
        if (!strcmp(pc->entry[i]->node_id, id)) return pc->entry[i];
    }
    return NULL;
}

static size_t prefix_index_locked(struct ds4_prefix_cache *pc,
                                  prefix_entry *entry) {
    for (size_t i = 0; i < pc->len; i++) {
        if (pc->entry[i] == entry) return i;
    }
    return SIZE_MAX;
}

static void prefix_add_locked(struct ds4_prefix_cache *pc, prefix_entry *e) {
    if (pc->len == pc->cap) {
        size_t cap = pc->cap ? pc->cap * 2u : 64u;
        pc->entry = kv_xrealloc(pc->entry, cap * sizeof(pc->entry[0]));
        pc->cap = cap;
    }
    pc->entry[pc->len++] = e;
}

static void prefix_remove_index_locked(struct ds4_prefix_cache *pc, size_t idx) {
    if (!pc || idx >= pc->len) return;
    prefix_entry *e = pc->entry[idx];
    prefix_entry *parent = prefix_find_id_locked(pc, e->parent_id);
    if (parent && parent->child_count) parent->child_count--;
    if (e->node) {
        if (pc->ram_bytes >= e->node_bytes) pc->ram_bytes -= e->node_bytes;
        else pc->ram_bytes = 0;
    }
    if (e->state == PREFIX_DURABLE) {
        if (pc->disk_bytes >= e->file_bytes) pc->disk_bytes -= e->file_bytes;
        else pc->disk_bytes = 0;
    } else if (e->state == PREFIX_WRITE_QUEUED) {
        if (pc->reserved_disk_bytes >= e->file_bytes)
            pc->reserved_disk_bytes -= e->file_bytes;
        else pc->reserved_disk_bytes = 0;
    }
    memmove(pc->entry + idx, pc->entry + idx + 1u,
            (pc->len - idx - 1u) * sizeof(pc->entry[0]));
    pc->len--;
    prefix_entry_free(e);
}

static bool prefix_victim_before(const prefix_entry *a,
                                 const prefix_entry *b) {
    if (!b) return true;
    if (a->last_used != b->last_used) return a->last_used < b->last_used;
    if (a->total_tokens != b->total_tokens)
        return a->total_tokens > b->total_tokens;
    return strcmp(a->node_id, b->node_id) < 0;
}

static void prefix_trim_ram_locked(struct ds4_prefix_cache *pc) {
    while (pc->reserved_ram_bytes > pc->ram_budget ||
           pc->ram_bytes > pc->ram_budget - pc->reserved_ram_bytes) {
        prefix_entry *victim = NULL;
        for (size_t i = 0; i < pc->len; i++) {
            prefix_entry *e = pc->entry[i];
            if (!e->node || e->pin_count != 0 ||
                e->state == PREFIX_WRITING ||
                e->state == PREFIX_WRITE_QUEUED)
                continue;
            if (e->state != PREFIX_DURABLE && e->child_count != 0) continue;
            if (prefix_victim_before(e, victim)) victim = e;
        }
        if (!victim) break;
        if (victim->state == PREFIX_DURABLE) {
            ds4_prefix_node_free(victim->node);
            victim->node = NULL;
            if (pc->ram_bytes >= victim->node_bytes)
                pc->ram_bytes -= victim->node_bytes;
            else
                pc->ram_bytes = 0;
        } else {
            const size_t idx = prefix_index_locked(pc, victim);
            if (idx == SIZE_MAX) break;
            prefix_remove_index_locked(pc, idx);
        }
    }
}

static uint64_t prefix_kvc_bytes(const ds4_kvstore *kc) {
    uint64_t total = 0;
    if (!kc || !kc->dir) return 0;
    DIR *dir = opendir(kc->dir);
    if (!dir) return 0;
    struct dirent *de;
    while ((de = readdir(dir)) != NULL) {
        char sha[41];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        char *path = ds4_kvstore_path_join(kc->dir, de->d_name);
        struct stat st;
        const bool ok = stat(path, &st) == 0 && st.st_size >= 0;
        free(path);
        if (!ok) continue;
        if (UINT64_MAX - total < (uint64_t)st.st_size) {
            total = UINT64_MAX;
            break;
        }
        total += (uint64_t)st.st_size;
    }
    closedir(dir);
    return total;
}

static void prefix_cancel_writes_locked(struct ds4_prefix_cache *pc,
                                        prefix_entry *root) {
    if (!pc || !root) return;
    for (size_t pass = 0; pass <= pc->len; pass++) {
        bool changed = false;
        for (size_t i = 0; i < pc->len; i++) {
            prefix_entry *e = pc->entry[i];
            bool dependent = e == root;
            if (!dependent && e->parent_id[0]) {
                prefix_entry *parent = prefix_find_id_locked(pc, e->parent_id);
                dependent = parent &&
                    (parent->state == PREFIX_RAM_ONLY ||
                     parent->state == PREFIX_FAILED);
            }
            if (!dependent || e->state == PREFIX_DURABLE ||
                e->state == PREFIX_RAM_ONLY || e->state == PREFIX_FAILED)
                continue;
            if (e->state == PREFIX_WRITE_QUEUED) {
                if (pc->reserved_disk_bytes >= e->file_bytes)
                    pc->reserved_disk_bytes -= e->file_bytes;
                else
                    pc->reserved_disk_bytes = 0;
            }
            e->state = e->node ? PREFIX_RAM_ONLY : PREFIX_FAILED;
            changed = true;
        }
        if (!changed) break;
    }
    pthread_cond_broadcast(&pc->cv);
}

static void prefix_mark_failed_locked(struct ds4_prefix_cache *pc,
                                      prefix_entry *root) {
    if (!pc || !root) return;
    root->state = PREFIX_FAILED;
    for (size_t pass = 0; pass <= pc->len; pass++) {
        bool changed = false;
        for (size_t i = 0; i < pc->len; i++) {
            prefix_entry *e = pc->entry[i];
            if (e->state == PREFIX_FAILED || !e->parent_id[0]) continue;
            prefix_entry *parent = prefix_find_id_locked(pc, e->parent_id);
            if (!parent || parent->state != PREFIX_FAILED) continue;
            if (e->state == PREFIX_WRITE_QUEUED) {
                if (pc->reserved_disk_bytes >= e->file_bytes)
                    pc->reserved_disk_bytes -= e->file_bytes;
                else
                    pc->reserved_disk_bytes = 0;
            }
            e->state = PREFIX_FAILED;
            changed = true;
        }
        if (!changed) break;
    }
    pthread_cond_broadcast(&pc->cv);
}

static void prefix_fill_file_header(uint8_t h[PREFIX_FILE_HEADER],
                                    const prefix_entry *e) {
    memset(h, 0, PREFIX_FILE_HEADER);
    memcpy(h, "PFX1", 4);
    ds4_kvstore_le_put32(h + 4, PREFIX_FILE_VERSION);
    prefix_put64(h + 8, e->compat_id);
    ds4_kvstore_le_put32(h + 16, e->parent_tokens);
    ds4_kvstore_le_put32(h + 20, e->total_tokens);
    ds4_kvstore_le_put32(h + 24, (uint32_t)e->text_len);
    prefix_put64(h + 28, e->node_bytes);
    prefix_put64(h + 36, e->last_used);
    memcpy(h + 44, e->parent_id, 40);
    memcpy(h + 84, e->prefix_sha, 40);
    memcpy(h + 124, e->node_id, 40);
    prefix_put64(h + 164, e->token_hash);
}

static bool prefix_parse_file_header(const uint8_t h[PREFIX_FILE_HEADER],
                                     prefix_entry *e) {
    if (memcmp(h, "PFX1", 4) ||
        ds4_kvstore_le_get32(h + 4) != PREFIX_FILE_VERSION)
        return false;
    e->compat_id = prefix_get64(h + 8);
    e->parent_tokens = ds4_kvstore_le_get32(h + 16);
    e->total_tokens = ds4_kvstore_le_get32(h + 20);
    e->text_len = ds4_kvstore_le_get32(h + 24);
    e->node_bytes = prefix_get64(h + 28);
    e->last_used = prefix_get64(h + 36);
    memcpy(e->parent_id, h + 44, 40);
    memcpy(e->prefix_sha, h + 84, 40);
    memcpy(e->node_id, h + 124, 40);
    e->token_hash = prefix_get64(h + 164);
    e->parent_id[40] = '\0';
    e->prefix_sha[40] = '\0';
    e->node_id[40] = '\0';
    if (e->compat_id == 0 || e->token_hash == 0 || e->total_tokens == 0 ||
        e->parent_tokens >= e->total_tokens ||
        e->text_len > UINT32_MAX || e->node_bytes == 0 ||
        e->node_bytes > SIZE_MAX)
        return false;
    e->file_bytes = PREFIX_FILE_HEADER + e->text_len + e->node_bytes;
    return e->file_bytes >= e->node_bytes;
}

static bool prefix_write_entry_file(prefix_entry *e) {
    if (!e || !e->path || !e->node) return false;
    kv_buf tmpb = {0};
    kv_buf_printf(&tmpb, "%s.tmp.%ld", e->path, (long)getpid());
    char *tmp = kv_buf_take(&tmpb);
    FILE *fp = fopen(tmp, "wb");
    if (!fp) {
        free(tmp);
        return false;
    }
    uint8_t h[PREFIX_FILE_HEADER];
    prefix_fill_file_header(h, e);
    bool ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
              fwrite(e->text, 1, e->text_len, fp) == e->text_len;
    char err[160] = {0};
    if (ok)
        ok = ds4_prefix_node_write(e->node, fp, err, sizeof(err)) == 0;
    if (ok) ok = fflush(fp) == 0 && fsync(fileno(fp)) == 0;
    if (fclose(fp) != 0) ok = false;
    if (ok) ok = rename(tmp, e->path) == 0;
    if (!ok) unlink(tmp);
    free(tmp);
    return ok;
}

static prefix_entry *prefix_writer_ready_locked(struct ds4_prefix_cache *pc) {
    for (size_t i = 0; i < pc->len; i++) {
        prefix_entry *e = pc->entry[i];
        if (e->state != PREFIX_WRITE_QUEUED) continue;
        if (!e->parent_id[0]) return e;
        prefix_entry *parent = prefix_find_id_locked(pc, e->parent_id);
        if (!parent || parent->state == PREFIX_FAILED ||
            parent->state == PREFIX_RAM_ONLY) {
            prefix_cancel_writes_locked(pc, e);
            continue;
        }
        if (parent->state == PREFIX_DURABLE) return e;
    }
    return NULL;
}

static bool prefix_disk_over_budget(struct ds4_prefix_cache *pc) {
    if (!pc || !pc->owner || pc->owner->budget_bytes == 0) return false;
    pthread_mutex_lock(&pc->mu);
    const uint64_t disk_bytes = pc->disk_bytes;
    const uint64_t reserved_disk_bytes = pc->reserved_disk_bytes;
    const uint64_t reserved_kvc_bytes = pc->reserved_kvc_bytes;
    pthread_mutex_unlock(&pc->mu);
    uint64_t total = prefix_kvc_bytes(pc->owner);
    if (UINT64_MAX - total < disk_bytes) return true;
    total += disk_bytes;
    if (UINT64_MAX - total < reserved_disk_bytes) return true;
    total += reserved_disk_bytes;
    if (UINT64_MAX - total < reserved_kvc_bytes) return true;
    total += reserved_kvc_bytes;
    return total > pc->owner->budget_bytes;
}

static bool prefix_gc_disk_one(struct ds4_prefix_cache *pc) {
    pthread_mutex_lock(&pc->mu);
    prefix_entry *victim = NULL;
    for (size_t i = 0; i < pc->len; i++) {
        prefix_entry *e = pc->entry[i];
        if ((e->state != PREFIX_DURABLE && e->state != PREFIX_FAILED) ||
            !e->path || e->pin_count != 0 ||
            e->child_count != 0)
            continue;
        if (prefix_victim_before(e, victim)) victim = e;
    }
    if (!victim) {
        pthread_mutex_unlock(&pc->mu);
        return false;
    }
    const bool failed = victim->state == PREFIX_FAILED;
    victim->state = PREFIX_WRITING;
    char *path = kv_xstrdup(victim->path);
    pthread_mutex_unlock(&pc->mu);

    const bool ok = unlink(path) == 0 || errno == ENOENT;
    free(path);
    pthread_mutex_lock(&pc->mu);
    if (ok) {
        if (pc->disk_bytes >= victim->file_bytes)
            pc->disk_bytes -= victim->file_bytes;
        else
            pc->disk_bytes = 0;
        if (failed) {
            size_t idx = prefix_index_locked(pc, victim);
            if (idx != SIZE_MAX) {
                victim->state = PREFIX_RAM_ONLY;
                prefix_remove_index_locked(pc, idx);
            }
        } else if (victim->node) {
            victim->state = PREFIX_RAM_ONLY;
        } else {
            size_t idx = prefix_index_locked(pc, victim);
            if (idx != SIZE_MAX) {
                /* file bytes were already removed from accounting */
                victim->state = PREFIX_RAM_ONLY;
                prefix_remove_index_locked(pc, idx);
            }
        }
    } else {
        victim->state = failed ? PREFIX_FAILED : PREFIX_DURABLE;
    }
    pthread_mutex_unlock(&pc->mu);
    return ok;
}

static bool prefix_cancel_newest_queued_leaf(struct ds4_prefix_cache *pc,
                                             const prefix_entry *writing) {
    pthread_mutex_lock(&pc->mu);
    prefix_entry *victim = NULL;
    for (size_t i = 0; i < pc->len; i++) {
        prefix_entry *e = pc->entry[i];
        if (e == writing || e->state != PREFIX_WRITE_QUEUED ||
            e->child_count != 0)
            continue;
        if (!victim || e->last_used > victim->last_used ||
            (e->last_used == victim->last_used &&
             e->total_tokens > victim->total_tokens))
            victim = e;
    }
    if (victim) prefix_cancel_writes_locked(pc, victim);
    pthread_mutex_unlock(&pc->mu);
    return victim != NULL;
}

static void *prefix_writer_main(void *arg) {
    struct ds4_prefix_cache *pc = arg;
    for (;;) {
        pthread_mutex_lock(&pc->mu);
        prefix_entry *e = NULL;
        while (!pc->stopping && !(e = prefix_writer_ready_locked(pc))) {
            pthread_cond_wait(&pc->cv, &pc->mu);
        }
        if (pc->stopping) {
            pthread_mutex_unlock(&pc->mu);
            break;
        }
        e->state = PREFIX_WRITING;
        e->pin_count++;
        pc->writer_busy = true;
        pthread_mutex_unlock(&pc->mu);

        bool fits = !pc->owner->budget_bytes ||
                    e->file_bytes <= pc->owner->budget_bytes;
        while (fits && prefix_disk_over_budget(pc)) {
            if (prefix_gc_disk_one(pc)) continue;
            if (prefix_cancel_newest_queued_leaf(pc, e)) continue;
            fits = false;
        }
        const bool ok = fits && prefix_write_entry_file(e);

        pthread_mutex_lock(&pc->mu);
        e->pin_count--;
        pc->writer_busy = false;
        if (pc->reserved_disk_bytes >= e->file_bytes)
            pc->reserved_disk_bytes -= e->file_bytes;
        else
            pc->reserved_disk_bytes = 0;
        if (ok) {
            e->state = PREFIX_DURABLE;
            pc->disk_bytes += e->file_bytes;
        } else {
            prefix_cancel_writes_locked(pc, e);
        }
        prefix_trim_ram_locked(pc);
        pthread_cond_broadcast(&pc->cv);
        pthread_mutex_unlock(&pc->mu);

    }
    return NULL;
}

static bool prefix_filename_id(const char *name, char id[41]) {
    if (!name || strlen(name) != 44 || strcmp(name + 40, ".ptn")) return false;
    for (int i = 0; i < 40; i++) {
        if (!isxdigit((unsigned char)name[i])) return false;
        id[i] = (char)tolower((unsigned char)name[i]);
    }
    id[40] = '\0';
    return true;
}

static void prefix_scan_disk(struct ds4_prefix_cache *pc) {
    DIR *dir = opendir(pc->dir);
    if (!dir) return;
    struct dirent *de;
    while ((de = readdir(dir)) != NULL) {
        const size_t name_len = strlen(de->d_name);
        if (name_len > 49 && !memcmp(de->d_name + 40, ".ptn.tmp.", 9)) {
            char base[46];
            memcpy(base, de->d_name, 45);
            base[45] = '\0';
            char stale_id[41];
            if (prefix_filename_id(base, stale_id)) {
                char *stale = ds4_kvstore_path_join(pc->dir, de->d_name);
                unlink(stale);
                free(stale);
            }
            continue;
        }
        char file_id[41];
        if (!prefix_filename_id(de->d_name, file_id)) continue;
        char *path = ds4_kvstore_path_join(pc->dir, de->d_name);
        FILE *fp = fopen(path, "rb");
        if (!fp) {
            free(path);
            continue;
        }
        uint8_t h[PREFIX_FILE_HEADER];
        prefix_entry *e = prefix_entry_new();
        bool ok = fread(h, 1, sizeof(h), fp) == sizeof(h) &&
                  prefix_parse_file_header(h, e) &&
                  !strcmp(file_id, e->node_id) &&
                  e->compat_id == ds4_engine_prefix_compat_id(pc->engine);
        struct stat st;
        if (ok) ok = fstat(fileno(fp), &st) == 0 &&
                     st.st_size >= 0 &&
                     (uint64_t)st.st_size == e->file_bytes;
        if (ok) {
            e->text = kv_xmalloc(e->text_len + 1u);
            ok = fread(e->text, 1, e->text_len, fp) == e->text_len;
            e->text[e->text_len] = '\0';
            char sha[41];
            if (ok) {
                ds4_kvstore_sha1_bytes_hex(e->text, e->text_len, sha);
                ok = !strcmp(sha, e->prefix_sha);
            }
        }
        fclose(fp);
        if (!ok || prefix_find_id_locked(pc, e->node_id)) {
            if (!ok) unlink(path);
            prefix_entry_free(e);
            free(path);
            continue;
        }
        e->path = path;
        e->state = PREFIX_DURABLE;
        pc->disk_bytes += e->file_bytes;
        if (e->last_used > pc->clock) pc->clock = e->last_used;
        prefix_add_locked(pc, e);
    }
    closedir(dir);

    bool removed;
    do {
        removed = false;
        for (size_t i = 0; i < pc->len;) {
            prefix_entry *e = pc->entry[i];
            if (e->parent_id[0] && !prefix_find_id_locked(pc, e->parent_id)) {
                unlink(e->path);
                prefix_remove_index_locked(pc, i);
                removed = true;
                continue;
            }
            i++;
        }
    } while (removed);
    for (size_t i = 0; i < pc->len; i++) {
        prefix_entry *parent =
            prefix_find_id_locked(pc, pc->entry[i]->parent_id);
        if (parent) parent->child_count++;
    }
}

bool ds4_kvstore_prefix_enable(ds4_kvstore *kc, ds4_engine *engine,
                               uint64_t memory_mb,
                               pthread_mutex_t *inference_mu) {
    if (!kc || !engine || memory_mb == 0 ||
        memory_mb > UINT64_MAX / (1024ull * 1024ull))
        return false;
    if (kc->prefix) return true;
    struct ds4_prefix_cache *pc = kv_xmalloc(sizeof(*pc));
    memset(pc, 0, sizeof(*pc));
    pc->owner = kc;
    pc->engine = engine;
    pc->inference_mu = inference_mu;
    pc->ram_budget = memory_mb * 1024ull * 1024ull;
    pthread_mutex_init(&pc->mu, NULL);
    pthread_cond_init(&pc->cv, NULL);
    kc->prefix = pc;

    if (kc->dir && kc->budget_bytes) {
        pc->dir = ds4_kvstore_path_join(kc->dir, "tree-v1");
        if (kv_mkdir_p(pc->dir)) {
            prefix_scan_disk(pc);
            if (pthread_create(&pc->writer, NULL,
                               prefix_writer_main, pc) == 0) {
                pc->writer_started = true;
            }
        }
    }
    kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
            "%s: immutable prefix tree enabled RAM=%llu MiB persistence=%s",
            kv_log_name(kc),
            (unsigned long long)memory_mb,
            pc->writer_started ? pc->dir : "off");
    return true;
}

bool ds4_kvstore_prefix_enabled(const ds4_kvstore *kc) {
    return kc && kc->prefix && kc->prefix->ram_budget != 0;
}

static void prefix_cache_destroy(ds4_kvstore *kc) {
    if (!kc || !kc->prefix) return;
    struct ds4_prefix_cache *pc = kc->prefix;
    if (pc->writer_started) ds4_kvstore_prefix_wait_idle(kc);
    pthread_mutex_lock(&pc->mu);
    pc->stopping = true;
    pthread_cond_broadcast(&pc->cv);
    pthread_mutex_unlock(&pc->mu);
    if (pc->writer_started) pthread_join(pc->writer, NULL);
    for (size_t i = 0; i < pc->len; i++) prefix_entry_free(pc->entry[i]);
    free(pc->entry);
    free(pc->dir);
    pthread_cond_destroy(&pc->cv);
    pthread_mutex_destroy(&pc->mu);
    free(pc);
    kc->prefix = NULL;
}

static int prefix_parent_compare(const void *ap, const void *bp) {
    const prefix_entry *a = *(prefix_entry *const *)ap;
    const prefix_entry *b = *(prefix_entry *const *)bp;
    if (a->text_len != b->text_len) return a->text_len < b->text_len ? 1 : -1;
    if (a->total_tokens != b->total_tokens)
        return a->total_tokens < b->total_tokens ? 1 : -1;
    return strcmp(a->node_id, b->node_id);
}

bool ds4_kvstore_prefix_capture(ds4_kvstore *kc, ds4_engine *engine,
                                ds4_session *session, const char *cache_text,
                                char *err, size_t errlen) {
    if (!ds4_kvstore_prefix_enabled(kc) || !engine || !session) return false;
    struct ds4_prefix_cache *pc = kc->prefix;
    const ds4_tokens *live = ds4_session_tokens(session);
    if (!live || live->len <= 0) return false;
    size_t text_len = 0;
    char *text = cache_text ? kv_xstrdup(cache_text) :
        ds4_kvstore_render_tokens_text(engine, live, &text_len);
    if (cache_text) text_len = strlen(cache_text);
    if (text_len == 0 || text_len > UINT32_MAX) {
        free(text);
        return false;
    }

    pthread_mutex_lock(&pc->mu);
    prefix_entry **parents =
        kv_xmalloc((pc->len ? pc->len : 1u) * sizeof(parents[0]));
    size_t parent_count = 0;
    for (size_t i = 0; i < pc->len; i++) {
        prefix_entry *e = pc->entry[i];
        if (!e->node || e->state == PREFIX_FAILED ||
            e->compat_id != ds4_engine_prefix_compat_id(engine) ||
            e->total_tokens >= (uint32_t)live->len ||
            e->text_len >= text_len ||
            !ds4_kvstore_byte_prefix_match(text, text_len,
                                           e->text, e->text_len))
            continue;
        e->pin_count++;
        parents[parent_count++] = e;
    }
    const uint64_t capture_reservation = ds4_session_payload_bytes(session);
    bool capture_reserved = false;
    bool room = capture_reservation <= pc->ram_budget &&
        UINT64_MAX - pc->reserved_ram_bytes >= capture_reservation;
    if (room) {
        pc->reserved_ram_bytes += capture_reservation;
        capture_reserved = true;
        prefix_trim_ram_locked(pc);
        room = pc->reserved_ram_bytes <= pc->ram_budget &&
               pc->ram_bytes <= pc->ram_budget - pc->reserved_ram_bytes;
    }
    if (!room) {
        if (capture_reserved &&
            pc->reserved_ram_bytes >= capture_reservation)
            pc->reserved_ram_bytes -= capture_reservation;
        for (size_t i = 0; i < parent_count; i++) parents[i]->pin_count--;
        pthread_mutex_unlock(&pc->mu);
        free(parents);
        free(text);
        if (err && errlen)
            snprintf(err, errlen,
                     "prefix capture exceeds the configured RAM budget");
        return false;
    }
    pthread_mutex_unlock(&pc->mu);
    qsort(parents, parent_count, sizeof(parents[0]), prefix_parent_compare);

    ds4_prefix_node *node = NULL;
    prefix_entry *chosen_parent = NULL;
    for (size_t i = 0; i < parent_count; i++) {
        if (ds4_session_prefix_node_capture(
                    session, parents[i]->node, &node, err, errlen) == 0) {
            chosen_parent = parents[i];
            break;
        }
    }
    if (!node && ds4_session_prefix_node_capture(
                     session, NULL, &node, err, errlen) != 0) {
        pthread_mutex_lock(&pc->mu);
        if (pc->reserved_ram_bytes >= capture_reservation)
            pc->reserved_ram_bytes -= capture_reservation;
        else
            pc->reserved_ram_bytes = 0;
        for (size_t i = 0; i < parent_count; i++) parents[i]->pin_count--;
        prefix_trim_ram_locked(pc);
        pthread_mutex_unlock(&pc->mu);
        free(parents);
        free(text);
        return false;
    }

    char node_id[41];
    prefix_node_id(chosen_parent ? chosen_parent->node_id : NULL,
                   node, node_id);
    char prefix_sha[41];
    ds4_kvstore_sha1_bytes_hex(text, text_len, prefix_sha);

    pthread_mutex_lock(&pc->mu);
    if (pc->reserved_ram_bytes >= capture_reservation)
        pc->reserved_ram_bytes -= capture_reservation;
    else
        pc->reserved_ram_bytes = 0;
    prefix_entry *existing = prefix_find_id_locked(pc, node_id);
    if (existing) {
        existing->last_used = ++pc->clock;
        for (size_t i = 0; i < parent_count; i++) parents[i]->pin_count--;
        pthread_mutex_unlock(&pc->mu);
        ds4_prefix_node_free(node);
        free(parents);
        free(text);
        return true;
    }

    prefix_entry *e = prefix_entry_new();
    memcpy(e->node_id, node_id, sizeof(e->node_id));
    memcpy(e->prefix_sha, prefix_sha, sizeof(e->prefix_sha));
    if (chosen_parent)
        memcpy(e->parent_id, chosen_parent->node_id, sizeof(e->parent_id));
    e->text = text;
    e->text_len = text_len;
    e->node = node;
    e->compat_id = ds4_prefix_node_compat_id(node);
    e->token_hash = ds4_prefix_node_token_hash(node);
    e->node_bytes = ds4_prefix_node_bytes(node);
    e->file_bytes = PREFIX_FILE_HEADER + text_len + e->node_bytes;
    e->parent_tokens = ds4_prefix_node_parent_tokens(node);
    e->total_tokens = ds4_prefix_node_total_tokens(node);
    e->last_used = ++pc->clock;
    if (pc->writer_started) {
        kv_buf name = {0};
        kv_buf_printf(&name, "%s.ptn", e->node_id);
        e->path = ds4_kvstore_path_join(pc->dir, name.ptr);
        free(name.ptr);
        e->state = PREFIX_WRITE_QUEUED;
        pc->reserved_disk_bytes += e->file_bytes;
    } else {
        e->state = PREFIX_RAM_ONLY;
    }
    prefix_add_locked(pc, e);
    pc->ram_bytes += e->node_bytes;
    if (chosen_parent) chosen_parent->child_count++;
    for (size_t i = 0; i < parent_count; i++) parents[i]->pin_count--;
    prefix_trim_ram_locked(pc);
    prefix_entry *admitted_entry = prefix_find_id_locked(pc, node_id);
    if (pc->ram_bytes > pc->ram_budget && admitted_entry &&
        admitted_entry->child_count == 0 && admitted_entry->pin_count == 0) {
        const size_t idx = prefix_index_locked(pc, admitted_entry);
        if (idx != SIZE_MAX) prefix_remove_index_locked(pc, idx);
        admitted_entry = NULL;
    }
    const bool admitted = admitted_entry != NULL &&
                          pc->ram_bytes <= pc->ram_budget;
    pthread_cond_broadcast(&pc->cv);
    pthread_mutex_unlock(&pc->mu);
    free(parents);
    if (!admitted) {
        if (err && errlen)
            snprintf(err, errlen,
                     "prefix node exceeds the configured RAM budget");
        return false;
    }
    return true;
}

static prefix_entry *prefix_best_locked(struct ds4_prefix_cache *pc,
                                        const char *prompt, size_t prompt_len,
                                        const ds4_tokens *prompt_tokens,
                                        const uint64_t *prompt_hashes) {
    prefix_entry *best = NULL;
    const uint64_t compat = ds4_engine_prefix_compat_id(pc->engine);
    for (size_t i = 0; i < pc->len; i++) {
        prefix_entry *e = pc->entry[i];
        if (e->state == PREFIX_FAILED || e->compat_id != compat ||
            e->text_len > prompt_len ||
            (prompt_tokens &&
             (e->total_tokens > (uint32_t)prompt_tokens->len ||
              !prompt_hashes ||
              e->token_hash != prompt_hashes[e->total_tokens])) ||
            !ds4_kvstore_byte_prefix_match(prompt, prompt_len,
                                           e->text, e->text_len))
            continue;
        if (!best || e->text_len > best->text_len ||
            (e->text_len == best->text_len &&
             e->total_tokens > best->total_tokens) ||
            (e->text_len == best->text_len &&
             e->total_tokens == best->total_tokens &&
             e->node && !best->node))
            best = e;
    }
    return best;
}

bool ds4_kvstore_prefix_find(ds4_kvstore *kc, const char *prompt_text,
                             const ds4_tokens *prompt_tokens,
                             ds4_prefix_lookup *out) {
    if (out) memset(out, 0, sizeof(*out));
    if (!ds4_kvstore_prefix_enabled(kc) || !prompt_text) return false;
    struct ds4_prefix_cache *pc = kc->prefix;
    uint64_t *prompt_hashes =
        prefix_prompt_token_hashes(prompt_tokens);
    pthread_mutex_lock(&pc->mu);
    prefix_entry *e = prefix_best_locked(pc, prompt_text, strlen(prompt_text),
                                         prompt_tokens, prompt_hashes);
    if (e && out) {
        out->tokens = (int)e->total_tokens;
        out->text_bytes = e->text_len;
        out->ram_resident = e->node != NULL;
        for (prefix_entry *p = e; p;
             p = prefix_find_id_locked(pc, p->parent_id)) {
            if (!p->node) {
                if (UINT64_MAX - out->restore_bytes < p->node_bytes) {
                    out->restore_bytes = UINT64_MAX;
                    break;
                }
                out->restore_bytes += p->node_bytes;
            }
            if (!p->parent_id[0]) break;
        }
    }
    pthread_mutex_unlock(&pc->mu);
    free(prompt_hashes);
    return e != NULL;
}

static bool prefix_load_node_file(prefix_entry *e, ds4_engine *engine,
                                  ds4_prefix_node **out,
                                  char *err, size_t errlen) {
    *out = NULL;
    FILE *fp = fopen(e->path, "rb");
    if (!fp) return false;
    bool ok = fseeko(fp, (off_t)(PREFIX_FILE_HEADER + e->text_len),
                     SEEK_SET) == 0 &&
              ds4_prefix_node_read(engine, fp, e->node_bytes,
                                   out, err, errlen) == 0;
    fclose(fp);
    if (ok) {
        char sha[41];
        prefix_node_id(e->parent_id, *out, sha);
        ok = !strcmp(sha, e->node_id);
    }
    if (!ok) {
        ds4_prefix_node_free(*out);
        *out = NULL;
    }
    return ok;
}

int ds4_kvstore_prefix_restore(ds4_kvstore *kc, ds4_engine *engine,
                               ds4_session *session, const char *prompt_text,
                               const ds4_tokens *prompt_tokens,
                               ds4_tokens *effective_prompt,
                               ds4_prefix_lookup *out,
                               char *err, size_t errlen) {
    if (out) memset(out, 0, sizeof(*out));
    if (!ds4_kvstore_prefix_enabled(kc) || !engine || !session ||
        !prompt_text)
        return 0;
    struct ds4_prefix_cache *pc = kc->prefix;
    uint64_t *prompt_hashes =
        prefix_prompt_token_hashes(prompt_tokens);
    pthread_mutex_lock(&pc->mu);
    prefix_entry *selected =
        prefix_best_locked(pc, prompt_text, strlen(prompt_text),
                           prompt_tokens, prompt_hashes);
    free(prompt_hashes);
    if (!selected) {
        pthread_mutex_unlock(&pc->mu);
        return 0;
    }
    const uint32_t selected_tokens = selected->total_tokens;
    const size_t selected_text_len = selected->text_len;
    prefix_entry **reverse =
        kv_xmalloc((pc->len ? pc->len : 1u) * sizeof(reverse[0]));
    size_t count = 0;
    for (prefix_entry *e = selected; e; e = prefix_find_id_locked(pc, e->parent_id)) {
        if (count >= pc->len || e->state == PREFIX_FAILED) break;
        e->pin_count++;
        reverse[count++] = e;
        if (!e->parent_id[0]) break;
    }
    if (count == 0 || reverse[count - 1u]->parent_id[0]) {
        for (size_t i = 0; i < count; i++) reverse[i]->pin_count--;
        pthread_mutex_unlock(&pc->mu);
        free(reverse);
        return 0;
    }
    prefix_entry **path = kv_xmalloc(count * sizeof(path[0]));
    for (size_t i = 0; i < count; i++) path[i] = reverse[count - i - 1u];
    uint64_t missing_bytes = 0;
    bool room = true;
    bool reserved = false;
    for (size_t i = 0; i < count; i++) {
        if (path[i]->node) continue;
        if (UINT64_MAX - missing_bytes < path[i]->node_bytes) {
            room = false;
            break;
        }
        missing_bytes += path[i]->node_bytes;
    }
    if (room && UINT64_MAX - pc->reserved_ram_bytes >= missing_bytes) {
        pc->reserved_ram_bytes += missing_bytes;
        reserved = true;
    } else {
        room = false;
    }
    if (room) {
        prefix_trim_ram_locked(pc);
        room = pc->reserved_ram_bytes <= pc->ram_budget &&
               pc->ram_bytes <= pc->ram_budget - pc->reserved_ram_bytes;
    }
    if (!room) {
        if (reserved && pc->reserved_ram_bytes >= missing_bytes)
            pc->reserved_ram_bytes -= missing_bytes;
        for (size_t i = 0; i < count; i++) path[i]->pin_count--;
        pthread_mutex_unlock(&pc->mu);
        free(path);
        free(reverse);
        if (err && errlen)
            snprintf(err, errlen,
                     "prefix restore chain exceeds the configured RAM budget");
        return 0;
    }
    pthread_mutex_unlock(&pc->mu);
    free(reverse);

    ds4_prefix_node **nodes = kv_xmalloc(count * sizeof(nodes[0]));
    bool *owned = kv_xmalloc(count * sizeof(owned[0]));
    memset(nodes, 0, count * sizeof(nodes[0]));
    memset(owned, 0, count * sizeof(owned[0]));
    bool ok = true;
    size_t failed_at = SIZE_MAX;
    for (size_t i = 0; ok && i < count; i++) {
        if (path[i]->node) {
            nodes[i] = path[i]->node;
        } else {
            ok = prefix_load_node_file(path[i], engine, &nodes[i],
                                       err, errlen);
            owned[i] = ok;
            if (!ok) failed_at = i;
        }
    }
    const double t0 = kv_now_sec();
    const bool inference_locked = ok && pc->inference_mu;
    if (inference_locked)
        pthread_mutex_lock(pc->inference_mu);
    if (ok)
        ok = ds4_session_prefix_chain_restore(
                    session, (const ds4_prefix_node *const *)nodes,
                    count, err, errlen) == 0;
    bool prompt_mismatch = false;
    if (ok && prompt_tokens) {
        const ds4_tokens *tokens = ds4_session_tokens(session);
        prompt_mismatch = !tokens || tokens->len != (int)selected_tokens ||
            prompt_tokens->len < (int)selected_tokens ||
            memcmp(tokens->v, prompt_tokens->v,
                   (size_t)selected_tokens * sizeof(tokens->v[0])) != 0;
        if (prompt_mismatch) {
            ds4_session_invalidate(session);
            if (err && errlen)
                snprintf(err, errlen,
                         "prefix tokenization does not match this prompt");
            ok = false;
        }
    }
    if (ok && effective_prompt && prompt_tokens) {
        ds4_tokens_copy(effective_prompt, prompt_tokens);
    } else if (ok && effective_prompt) {
        const ds4_tokens *tokens = ds4_session_tokens(session);
        ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
                engine, tokens, prompt_text + selected_text_len,
                effective_prompt);
    }
    if (inference_locked)
        pthread_mutex_unlock(pc->inference_mu);

    pthread_mutex_lock(&pc->mu);
    if (pc->reserved_ram_bytes >= missing_bytes)
        pc->reserved_ram_bytes -= missing_bytes;
    else
        pc->reserved_ram_bytes = 0;
    if (ok) {
        const uint64_t used = ++pc->clock;
        for (size_t i = 0; i < count; i++) {
            path[i]->last_used = used;
            if (owned[i] && !path[i]->node) {
                path[i]->node = nodes[i];
                pc->ram_bytes += path[i]->node_bytes;
                owned[i] = false;
            }
        }
    } else if (!prompt_mismatch) {
        prefix_mark_failed_locked(
            pc, failed_at == SIZE_MAX ? selected : path[failed_at]);
    }
    for (size_t i = 0; i < count; i++) path[i]->pin_count--;
    prefix_trim_ram_locked(pc);
    if (ok && out) {
        out->tokens = (int)selected->total_tokens;
        out->text_bytes = selected_text_len;
        out->ram_resident = selected->node != NULL;
    }
    pthread_cond_broadcast(&pc->cv);
    pthread_mutex_unlock(&pc->mu);

    for (size_t i = 0; i < count; i++) {
        if (owned[i]) ds4_prefix_node_free(nodes[i]);
    }
    free(owned);
    free(nodes);
    free(path);
    if (!ok) return 0;
    kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
            "%s: prefix tree hit tokens=%u text=%zu load=%.1f ms",
            kv_log_name(kc), selected_tokens, selected_text_len,
            (kv_now_sec() - t0) * 1000.0);
    return (int)selected_tokens;
}

void ds4_kvstore_prefix_wait_idle(ds4_kvstore *kc) {
    if (!kc || !kc->prefix) return;
    struct ds4_prefix_cache *pc = kc->prefix;
    pthread_mutex_lock(&pc->mu);
    for (;;) {
        bool pending = pc->writer_busy;
        for (size_t i = 0; !pending && i < pc->len; i++) {
            pending = pc->entry[i]->state == PREFIX_WRITE_QUEUED ||
                      pc->entry[i]->state == PREFIX_WRITING;
        }
        if (!pending) break;
        pthread_cond_wait(&pc->cv, &pc->mu);
    }
    pthread_mutex_unlock(&pc->mu);
}

void ds4_kvstore_prefix_stats(ds4_kvstore *kc, ds4_prefix_stats *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!kc || !kc->prefix) return;
    struct ds4_prefix_cache *pc = kc->prefix;
    pthread_mutex_lock(&pc->mu);
    out->nodes = pc->len;
    out->ram_bytes = pc->ram_bytes;
    out->disk_bytes = pc->disk_bytes;
    for (size_t i = 0; i < pc->len; i++) {
        prefix_entry *e = pc->entry[i];
        if (e->node) out->ram_nodes++;
        if (e->state == PREFIX_DURABLE) out->durable_nodes++;
        if (e->state == PREFIX_WRITE_QUEUED ||
            e->state == PREFIX_WRITING)
            out->pending_nodes++;
    }
    pthread_mutex_unlock(&pc->mu);
}
