#ifndef INTERNAL_EXTRACTOR_PROBE_H
#define INTERNAL_EXTRACTOR_PROBE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HYPTCN_PAGE_SIZE 4096

int fetch_guest_page(const char* vm_name, uint64_t physical_address, unsigned char* buffer);

#ifdef __cplusplus
}
#endif

#endif // INTERNAL_EXTRACTOR_PROBE_H
