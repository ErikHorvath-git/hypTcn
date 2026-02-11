#include "probe.h"
#include <stddef.h>

int fetch_ram_page(uint64_t address, char* buffer) {
    if (buffer == NULL) {
        return -1;
    }

    for (size_t i = 0; i < HYPTCN_PAGE_SIZE; i++) {
        buffer[i] = (char)((address + i) & 0xFF);
    }

    return 0;
}
