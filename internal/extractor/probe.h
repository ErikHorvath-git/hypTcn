#ifndef INTERNAL_EXTRACTOR_PROBE_H
#define INTERNAL_EXTRACTOR_PROBE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HYPTCN_PAGE_SIZE      4096
#define HYPTCN_MAX_PROCS      256
#define HYPTCN_MAX_MODS       128
#define HYPTCN_MAX_CONNS      512
#define HYPTCN_PROC_NAME_LEN  16   /* TASK_COMM_LEN */
#define HYPTCN_MOD_NAME_LEN   56   /* MODULE_NAME_LEN - padding */

/* Opaque handle returned by hyptcn_vmi_open / hyptcn_vmi_open_with_sysmap. */
typedef struct hyptcn_vmi_handle hyptcn_vmi_handle_t;

/* Per-process record returned by hyptcn_get_process_list. */
typedef struct {
    int32_t  pid;
    char     name[HYPTCN_PROC_NAME_LEN];
    uint64_t cr3;   /* page-table base (physical address) */
} hyptcn_proc_t;

/* Per-module record returned by hyptcn_get_kernel_modules. */
typedef struct {
    char     name[HYPTCN_MOD_NAME_LEN];
    uint64_t base_address;
    uint64_t size;
} hyptcn_module_t;

/* Per-connection record returned by hyptcn_get_network_connections. */
typedef struct {
    uint32_t local_ip;
    uint16_t local_port;
    uint32_t remote_ip;
    uint16_t remote_port;
    uint8_t  state;   /* TCP_ESTABLISHED=1 … TCP_CLOSE=7 */
} hyptcn_conn_t;

/* ── connection lifecycle ──────────────────────────────────────────────────── */

/*
 * Open a persistent libvmi connection using raw physical-memory mode only.
 * Returns NULL on failure.
 */
hyptcn_vmi_handle_t *hyptcn_vmi_open(const char *vm_name);

/*
 * Open a persistent libvmi connection with OS-layer introspection enabled.
 * Uses vmi_init_complete() + the supplied System.map for Linux symbol
 * resolution.  Falls back transparently to hyptcn_vmi_open() when
 * sysmap_path is NULL or vmi_init_complete() fails.
 * Returns NULL on failure.
 */
hyptcn_vmi_handle_t *hyptcn_vmi_open_with_sysmap(const char *vm_name,
                                                   const char *sysmap_path);

/*
 * Returns 1 if the handle was opened with vmi_init_complete (OS-layer active),
 * 0 otherwise.
 */
int hyptcn_has_os_layer(const hyptcn_vmi_handle_t *handle);

/*
 * Read exactly HYPTCN_PAGE_SIZE bytes from physical_address into buffer.
 * Returns 0 on success, non-zero on error.
 */
int hyptcn_read_page(hyptcn_vmi_handle_t *handle,
                     uint64_t physical_address,
                     unsigned char *buffer);

/*
 * Destroy the libvmi connection and free the handle.
 * Safe to call with NULL.
 */
void hyptcn_vmi_close(hyptcn_vmi_handle_t *handle);

/* ── OS-layer introspection (require os_layer_ready == 1) ─────────────────── */

/*
 * Walk the Linux task_struct linked list starting from init_task.
 * Fills out_procs[0..return_value-1].  Returns process count, or 0 if the
 * OS layer is not active or the walk fails.
 */
int hyptcn_get_process_list(hyptcn_vmi_handle_t *handle,
                             hyptcn_proc_t *out_procs,
                             int max_procs);

/*
 * Walk the Linux kernel module linked list (struct module.list via the
 * 'modules' kernel symbol).
 * Returns module count, or 0 on failure / no OS layer.
 */
int hyptcn_get_kernel_modules(hyptcn_vmi_handle_t *handle,
                               hyptcn_module_t *out_modules,
                               int max_modules);

/*
 * Walk the TCP established-connection hash table (tcp_hashinfo.ehash).
 * Returns connection count, or 0 on failure / no OS layer.
 */
int hyptcn_get_network_connections(hyptcn_vmi_handle_t *handle,
                                    hyptcn_conn_t *out_conns,
                                    int max_conns);

/*
 * Translate a virtual address in the address space identified by dtb (CR3)
 * to its physical address using libvmi's page-table walker.
 * Returns 0 on success, non-zero on error.
 */
int hyptcn_translate_v2p(hyptcn_vmi_handle_t *handle,
                          uint64_t dtb,
                          uint64_t vaddr,
                          uint64_t *paddr_out);

#ifdef __cplusplus
}
#endif

#endif /* INTERNAL_EXTRACTOR_PROBE_H */
