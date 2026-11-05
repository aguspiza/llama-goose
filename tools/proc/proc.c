// A child process for llama-cli's server, and Ctrl+C. The child runs in a
// process group of its own, so that the console's Ctrl+C goes to the CLI
// alone, and (on Windows) in a job that ends it when the CLI ends.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char * data; int64_t len; } lg_buf;            // a u8[:]
typedef struct { const char * data; int64_t len; } lg_text;     // a const u8[:]

static volatile int64_t lg_interrupts = 0;

#ifdef _WIN32
#include <windows.h>

static BOOL WINAPI lg_ctrl(DWORD type) {
    if (type != CTRL_C_EVENT && type != CTRL_BREAK_EVENT) return FALSE;
    if (lg_interrupts > 0) {
        // a second Ctrl+C: out at once, the colors reset
        fputs("\033[0m\n", stdout);
        fflush(stdout);
        ExitProcess(130);
    }
    lg_interrupts++;
    return TRUE;
}

void proc_catch_interrupts(void) { SetConsoleCtrlHandler(lg_ctrl, TRUE); }

static HANDLE lg_job = NULL;

// Starts cmdline (UTF-8, one string) with its output discarded: a handle,
// or 0.
int64_t proc_spawn(lg_text cmdline) {
    int n = MultiByteToWideChar(CP_UTF8, 0, cmdline.data, (int) cmdline.len, NULL, 0);
    wchar_t * w = (wchar_t *) calloc((size_t) n + 1, sizeof(wchar_t));
    if (!w) return 0;
    MultiByteToWideChar(CP_UTF8, 0, cmdline.data, (int) cmdline.len, w, n);
    SECURITY_ATTRIBUTES sa = { sizeof sa, NULL, TRUE };
    HANDLE nul = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, NULL);
    STARTUPINFOW si;
    memset(&si, 0, sizeof si);
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = nul;
    si.hStdError = nul;
    PROCESS_INFORMATION pi;
    BOOL ok = CreateProcessW(NULL, w, NULL, NULL, TRUE, CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED, NULL, NULL, &si, &pi);
    free(w);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!ok) return 0;
    if (!lg_job) {
        lg_job = CreateJobObjectW(NULL, NULL);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION info;
        memset(&info, 0, sizeof info);
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(lg_job, JobObjectExtendedLimitInformation, &info, sizeof info);
    }
    AssignProcessToJobObject(lg_job, pi.hProcess);
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    return (int64_t) (intptr_t) pi.hProcess;
}

int64_t proc_alive(int64_t h) {
    DWORD code = 0;
    return GetExitCodeProcess((HANDLE) (intptr_t) h, &code) && code == STILL_ACTIVE ? 1 : 0;
}

void proc_kill(int64_t h) {
    TerminateProcess((HANDLE) (intptr_t) h, 1);
    WaitForSingleObject((HANDLE) (intptr_t) h, 5000);
    CloseHandle((HANDLE) (intptr_t) h);
}

// This program's path, in UTF-8: its length (0 on failure).
int64_t proc_exe_path(lg_buf out) {
    wchar_t w[4096];
    DWORD n = GetModuleFileNameW(NULL, w, 4096);
    if (n == 0 || n >= 4096) return 0;
    return WideCharToMultiByte(CP_UTF8, 0, w, (int) n, out.data, (int) out.len, NULL, NULL);
}

#else
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

static void lg_sigint(int sig) {
    (void) sig;
    if (lg_interrupts > 0) {
        const char reset[] = "\033[0m\n";
        (void) !write(1, reset, sizeof reset - 1);
        _exit(130);
    }
    lg_interrupts++;
}

void proc_catch_interrupts(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = lg_sigint;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
}

// Starts cmdline through /bin/sh, its output discarded: the pid, or 0.
int64_t proc_spawn(lg_text cmdline) {
    char * c = (char *) malloc((size_t) cmdline.len + 1);
    if (!c) return 0;
    memcpy(c, cmdline.data, (size_t) cmdline.len);
    c[cmdline.len] = 0;
    pid_t pid = fork();
    if (pid == 0) {
        setpgid(0, 0);
        int nul = open("/dev/null", O_WRONLY);
        if (nul >= 0) { dup2(nul, 1); dup2(nul, 2); }
        execl("/bin/sh", "sh", "-c", c, (char *) NULL);
        _exit(127);
    }
    free(c);
    return pid > 0 ? (int64_t) pid : 0;
}

int64_t proc_alive(int64_t h) {
    int status;
    return waitpid((pid_t) h, &status, WNOHANG) == 0 ? 1 : 0;
}

void proc_kill(int64_t h) {
    kill((pid_t) h, SIGTERM);
    waitpid((pid_t) h, NULL, 0);
}

int64_t proc_exe_path(lg_buf out) {
    ssize_t n = readlink("/proc/self/exe", out.data, (size_t) out.len);
    return n > 0 ? (int64_t) n : 0;
}
#endif

// How many Ctrl+C came since the last take, and none from then on.
int64_t proc_take_interrupts(void) {
    int64_t n = lg_interrupts;
    lg_interrupts = 0;
    return n;
}

int64_t proc_interrupts(void) { return lg_interrupts; }
