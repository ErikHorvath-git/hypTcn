#include "probe.h"
#include <libvmi/libvmi.h>
#include <arpa/inet.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── internal handle struct ────────────────────────────────────────────────── */

struct hyptcn_vmi_handle {
    vmi_instance_t vmi;
    int os_layer_ready; /* 1 = vmi_init_complete with System.map succeeded */
};

/* ── helpers ────────────────────────────────────────────────────────────────── */

/*
 * Safe field read: return the libvmi offset name, or 0 if unavailable.
 * A return value of 0 is used as a sentinel to skip reads when the OS
 * profile was not loaded (os_layer_ready == 0).
 */
static addr_t _off(vmi_instance_t vmi, const char *name)
{
    addr_t val = 0;
    vmi_get_offset(vmi, name, &val);
    return val;
}

/* ── connection lifecycle ───────────────────────────────────────────────────── */

hyptcn_vmi_handle_t *hyptcn_vmi_open(const char *vm_name)
{
    if (vm_name == NULL)
        return NULL;

    hyptcn_vmi_handle_t *h = malloc(sizeof(*h));
    if (h == NULL)
        return NULL;
    h->vmi = NULL;
    h->os_layer_ready = 0;

    /*
     * Use vmi_init (not vmi_init_complete) — we only need raw physical memory
     * reads via vmi_read_pa, so OS-layer init (System.map / rekall profile)
     * is not required here.
     */
    status_t st = vmi_init(
        &h->vmi,
        VMI_KVM,
        vm_name,
        VMI_INIT_DOMAINNAME,
        NULL,
        NULL
    );

    if (st == VMI_FAILURE) {
        fprintf(stderr, "hyptcn: vmi_init failed for \"%s\"\n", vm_name);
        free(h);
        return NULL;
    }

    return h;
}

hyptcn_vmi_handle_t *hyptcn_vmi_open_with_sysmap(const char *vm_name,
                                                   const char *sysmap_path)
{
    if (vm_name == NULL)
        return NULL;

    /* NULL sysmap_path → fall back to raw mode. */
    if (sysmap_path == NULL)
        return hyptcn_vmi_open(vm_name);

    hyptcn_vmi_handle_t *h = malloc(sizeof(*h));
    if (h == NULL)
        return NULL;
    h->vmi = NULL;
    h->os_layer_ready = 0;

    /*
     * Build a VMI_CONFIG_STRING for vmi_init_complete.
     * Format: "{ os = linux; sysmap = /path/to/System.map; }"
     */
    char config[2048];
    snprintf(config, sizeof(config),
             "{ os = linux; sysmap = %s; }",
             sysmap_path);

    status_t st = vmi_init_complete(
        &h->vmi,
        vm_name,
        VMI_INIT_DOMAINNAME,
        NULL,               /* vmi_init_data_t — no extra init data */
        VMI_CONFIG_STRING,
        config,
        NULL                /* vmi_arch_interface_t — use default */
    );

    if (st == VMI_FAILURE) {
        fprintf(stderr,
                "hyptcn: vmi_init_complete failed for \"%s\" (sysmap=%s); "
                "falling back to raw mode\n",
                vm_name, sysmap_path);
        free(h);
        return hyptcn_vmi_open(vm_name);
    }

    h->os_layer_ready = 1;
    fprintf(stderr, "hyptcn: OS-layer active for \"%s\" (sysmap=%s)\n",
            vm_name, sysmap_path);
    return h;
}

int hyptcn_has_os_layer(const hyptcn_vmi_handle_t *handle)
{
    return (handle != NULL) ? handle->os_layer_ready : 0;
}

int hyptcn_read_page(hyptcn_vmi_handle_t *handle,
                     uint64_t physical_address,
                     unsigned char *buffer)
{
    if (handle == NULL || buffer == NULL)
        return -1;

    status_t st = vmi_read_pa(handle->vmi, physical_address,
                               HYPTCN_PAGE_SIZE, buffer, NULL);
    return (st == VMI_SUCCESS) ? 0 : -2;
}

void hyptcn_vmi_close(hyptcn_vmi_handle_t *handle)
{
    if (handle == NULL)
        return;
    vmi_destroy(handle->vmi);
    free(handle);
}

/* ── OS-layer introspection ─────────────────────────────────────────────────── */

/*
 * Walk the Linux task_struct doubly-linked list starting at init_task.
 *
 * libvmi resolves struct offsets via the loaded System.map + kernel DWARF or
 * a rekall/volatility profile.  The offset names used here ("task_pid",
 * "task_name", "task_tasks", "task_mm", "mm_pgd") are the canonical names
 * registered in libvmi's OS configuration for Linux guests.
 */
int hyptcn_get_process_list(hyptcn_vmi_handle_t *h,
                             hyptcn_proc_t *out,
                             int max)
{
    if (!h || !out || max <= 0 || !h->os_layer_ready)
        return 0;

    addr_t init_task_va = 0;
    if (vmi_translate_ksym2v(h->vmi, "init_task", &init_task_va) != VMI_SUCCESS) {
        fprintf(stderr, "hyptcn: cannot resolve init_task symbol\n");
        return 0;
    }

    addr_t tasks_off = _off(h->vmi, "task_tasks");
    addr_t pid_off   = _off(h->vmi, "task_pid");
    addr_t name_off  = _off(h->vmi, "task_name");
    addr_t mm_off    = _off(h->vmi, "task_mm");

    if (tasks_off == 0 || pid_off == 0) {
        fprintf(stderr, "hyptcn: task_struct offsets unavailable\n");
        return 0;
    }

    int count = 0;
    addr_t cur_task = init_task_va;

    do {
        /* Read PID. */
        uint32_t pid = 0;
        vmi_read_32_va(h->vmi, cur_task + pid_off, 0, &pid);

        /* Read process name (comm[TASK_COMM_LEN = 16]). */
        memset(out[count].name, 0, HYPTCN_PROC_NAME_LEN);
        if (name_off != 0) {
            vmi_read_va(h->vmi, cur_task + name_off, 0,
                        HYPTCN_PROC_NAME_LEN - 1, out[count].name, NULL);
        }
        out[count].pid = (int32_t)pid;

        /* Read CR3 (page-table base) via vmi_pid_to_dtb. */
        out[count].cr3 = 0;
        addr_t dtb = 0;
        if (mm_off != 0 &&
            vmi_pid_to_dtb(h->vmi, (vmi_pid_t)pid, &dtb) == VMI_SUCCESS) {
            out[count].cr3 = (uint64_t)dtb;
        }

        count++;

        /* Follow tasks.next; subtract tasks_off to recover task_struct base. */
        addr_t next_list_head = 0;
        if (vmi_read_addr_va(h->vmi, cur_task + tasks_off, 0,
                             &next_list_head) != VMI_SUCCESS)
            break;
        cur_task = next_list_head - tasks_off;

    } while (cur_task != init_task_va && count < max);

    return count;
}

/*
 * Walk the kernel module list.
 *
 * The 'modules' symbol is a list_head that is the head of the intrusive
 * doubly-linked list of struct module objects.  Each struct module embeds
 * a list_head 'list' member (conventionally at a small offset) and the
 * module's name in a char array.
 *
 * Offsets used:
 *   "module_list"       — offset of list_head within struct module
 *   "module_name"       — offset of name[] within struct module
 *   "module_core"       — offset of core_layout.base (load address)
 *   "module_core_size"  — offset of core_layout.size
 */
int hyptcn_get_kernel_modules(hyptcn_vmi_handle_t *h,
                               hyptcn_module_t *out,
                               int max)
{
    if (!h || !out || max <= 0 || !h->os_layer_ready)
        return 0;

    addr_t modules_va = 0;
    if (vmi_translate_ksym2v(h->vmi, "modules", &modules_va) != VMI_SUCCESS) {
        fprintf(stderr, "hyptcn: cannot resolve modules symbol\n");
        return 0;
    }

    addr_t list_off      = _off(h->vmi, "module_list");
    addr_t name_off      = _off(h->vmi, "module_name");
    addr_t core_off      = _off(h->vmi, "module_core");
    addr_t core_size_off = _off(h->vmi, "module_core_size");

    if (list_off == 0 || name_off == 0) {
        fprintf(stderr, "hyptcn: module offsets unavailable\n");
        return 0;
    }

    int count = 0;

    /* modules is a list_head — read its .next pointer. */
    addr_t cur_list = 0;
    if (vmi_read_addr_va(h->vmi, modules_va, 0, &cur_list) != VMI_SUCCESS)
        return 0;

    while (cur_list != 0 && cur_list != modules_va && count < max) {
        addr_t mod_base = cur_list - list_off;

        /* Read module name. */
        memset(out[count].name, 0, HYPTCN_MOD_NAME_LEN);
        vmi_read_va(h->vmi, mod_base + name_off, 0,
                    HYPTCN_MOD_NAME_LEN - 1, out[count].name, NULL);

        /* Read load address and size from core_layout. */
        out[count].base_address = 0;
        out[count].size         = 0;
        if (core_off != 0) {
            addr_t core_base = 0;
            vmi_read_addr_va(h->vmi, mod_base + core_off, 0, &core_base);
            out[count].base_address = (uint64_t)core_base;
        }
        if (core_size_off != 0) {
            uint32_t core_sz = 0;
            vmi_read_32_va(h->vmi, mod_base + core_size_off, 0, &core_sz);
            out[count].size = (uint64_t)core_sz;
        }

        count++;

        /* Follow list->next. */
        addr_t next = 0;
        if (vmi_read_addr_va(h->vmi, cur_list, 0, &next) != VMI_SUCCESS)
            break;
        cur_list = next;
    }

    return count;
}

/*
 * Walk the TCP established-connection hash table.
 *
 * Linux stores TCP sockets in tcp_hashinfo.ehash, an array of
 * inet_ehash_bucket structs.  Each bucket holds a hlist_nulls_head whose
 * first pointer chains sock/inet_sock objects.  A pointer with its low bit
 * set is a nulls marker (end of chain).
 *
 * The offsets for inet_sock fields (saddr, daddr, sport, dport) and for
 * tcp_hashinfo itself depend on the kernel version.  We use vmi_get_offset()
 * when the OS profile provides them; otherwise we fall back to well-known
 * values for Linux 5.15 x86-64.  A production deployment would supply a
 * rekall profile for exact offsets.
 *
 * We scan at most 4096 hash buckets to bound execution time.
 */
int hyptcn_get_network_connections(hyptcn_vmi_handle_t *h,
                                    hyptcn_conn_t *out,
                                    int max)
{
    if (!h || !out || max <= 0 || !h->os_layer_ready)
        return 0;

    addr_t tcp_hashinfo_va = 0;
    if (vmi_translate_ksym2v(h->vmi, "tcp_hashinfo", &tcp_hashinfo_va)
            != VMI_SUCCESS) {
        fprintf(stderr, "hyptcn: cannot resolve tcp_hashinfo symbol\n");
        return 0;
    }

    /*
     * struct inet_hashinfo (Linux 5.x):
     *   offset 0:   ehash      — pointer to inet_ehash_bucket array
     *   offset 16:  ehash_mask — (n_buckets - 1)
     * These are approximate; verified on Linux 5.15 x86-64.
     */
    addr_t ehash_va  = 0;
    uint32_t ehash_mask = 0;
    vmi_read_addr_va(h->vmi, tcp_hashinfo_va + 0,  0, &ehash_va);
    vmi_read_32_va  (h->vmi, tcp_hashinfo_va + 16, 0, &ehash_mask);

    if (ehash_va == 0 || ehash_mask == 0)
        return 0;

    uint32_t n_buckets = (ehash_mask + 1 < 4096) ? (ehash_mask + 1) : 4096;

    /*
     * struct sock (inet_sock) field offsets for IPv4 TCP (Linux 5.15 x86-64):
     *   sk_state  +12  (u8)
     *   sk_daddr  +68  (u32, remote IP, network byte order)
     *   sk_rcv_saddr +76 (u32, local IP, network byte order)
     * struct inet_sock:
     *   inet_sport +360 (u16, local port, host byte order in kernel)
     *   inet_dport +362 (u16, remote port, network byte order)
     *
     * These offsets are fragile across kernel versions.  A rekall profile
     * would provide exact values via vmi_get_offset().
     */
    addr_t off_state = _off(h->vmi, "sock_state");
    addr_t off_daddr = _off(h->vmi, "sock_daddr");
    addr_t off_saddr = _off(h->vmi, "sock_rcv_saddr");
    addr_t off_sport = _off(h->vmi, "inet_sport");
    addr_t off_dport = _off(h->vmi, "inet_dport");

    /* Fall back to Linux 5.15 x86-64 hardcoded values when profile absent. */
    if (off_state == 0) off_state = 12;
    if (off_daddr == 0) off_daddr = 68;
    if (off_saddr == 0) off_saddr = 76;
    if (off_sport == 0) off_sport = 360;
    if (off_dport == 0) off_dport = 362;

    int count = 0;

    for (uint32_t b = 0; b < n_buckets && count < max; b++) {
        /* Each inet_ehash_bucket is one pointer (the hlist_nulls_head). */
        addr_t bucket_va = ehash_va + (addr_t)b * sizeof(addr_t);
        addr_t node_va   = 0;

        if (vmi_read_addr_va(h->vmi, bucket_va, 0, &node_va) != VMI_SUCCESS)
            continue;

        /* Walk the nulls-list chain (low bit set = end marker). */
        while (node_va && !(node_va & 1) && count < max) {
            /* The hlist_nulls_node is the first member of struct sock. */
            addr_t sock_va = node_va;

            uint8_t  state = 0;
            uint32_t daddr = 0, saddr = 0;
            uint16_t dport = 0, sport = 0;

            vmi_read_8_va (h->vmi, sock_va + off_state, 0, &state);
            vmi_read_32_va(h->vmi, sock_va + off_daddr, 0, &daddr);
            vmi_read_32_va(h->vmi, sock_va + off_saddr, 0, &saddr);
            vmi_read_16_va(h->vmi, sock_va + off_sport, 0, &sport);
            vmi_read_16_va(h->vmi, sock_va + off_dport, 0, &dport);

            /* Record non-trivial sockets (at least one endpoint non-zero). */
            if (daddr != 0 || saddr != 0) {
                out[count].remote_ip   = daddr;
                out[count].local_ip    = saddr;
                /* Ports in kernel are network byte order for dport, host for sport */
                out[count].remote_port = ntohs(dport);
                out[count].local_port  = sport;
                out[count].state       = state;
                count++;
            }

            /* Follow the next pointer in the hlist_nulls_node. */
            addr_t next_node = 0;
            if (vmi_read_addr_va(h->vmi, node_va, 0, &next_node) != VMI_SUCCESS)
                break;
            node_va = next_node;
        }
    }

    return count;
}

/*
 * Translate a virtual address in the process identified by dtb (CR3) to its
 * physical address using libvmi's page-table walker.
 */
int hyptcn_translate_v2p(hyptcn_vmi_handle_t *h,
                          uint64_t dtb,
                          uint64_t vaddr,
                          uint64_t *paddr_out)
{
    if (!h || !paddr_out)
        return -1;

    addr_t pa = 0;
    status_t st = vmi_pagetable_lookup(h->vmi, (addr_t)dtb, (addr_t)vaddr, &pa);
    if (st != VMI_SUCCESS)
        return -2;

    *paddr_out = (uint64_t)pa;
    return 0;
}
