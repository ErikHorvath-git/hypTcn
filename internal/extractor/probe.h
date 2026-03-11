#ifndef INTERNAL_EXTRACTOR_PROBE_H
#define INTERNAL_EXTRACTOR_PROBE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HYPTCN_PAGE_SIZE 4096

/* Opaque handle returned by hyptcn_vmi_open. */
typedef struct hyptcn_vmi_handle hyptcn_vmi_handle_t;

/*
 * Open a persistent libvmi connection to the named KVM guest.
 * Returns NULL on failure; the caller owns the handle and must call
 * hyptcn_vmi_close() when done.
 */
hyptcn_vmi_handle_t *hyptcn_vmi_open(const char *vm_name);

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

#ifdef __cplusplus
}
#endif

#endif /* INTERNAL_EXTRACTOR_PROBE_H */
