#include "probe.h"
#include <libvmi/libvmi.h>
#include <stddef.h>

#define HYPTCN_INIT_ERROR -1
#define HYPTCN_READ_ERROR -2

int fetch_guest_page(const char* vm_name, uint64_t physical_address, unsigned char* buffer) {
    if (vm_name == NULL || buffer == NULL) {
        return HYPTCN_INIT_ERROR;
    }

    vmi_instance_t vmi = NULL;
    if (vmi_init(&vmi, VMI_KVM, vm_name, VMI_INIT_DOMAINNAME, NULL, NULL) == VMI_FAILURE) {
        return HYPTCN_INIT_ERROR;
    }

    status_t status = vmi_read_pa(vmi, physical_address, HYPTCN_PAGE_SIZE, buffer, NULL);
    vmi_destroy(vmi);

    if (status != VMI_SUCCESS) {
        return HYPTCN_READ_ERROR;
    }

    return 0;
}
