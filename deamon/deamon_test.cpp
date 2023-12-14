//daemon.c
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/fs.h>
#include <linux/limits.h>


int _daemon(int nochdir, int noclose) {
 pid_t pid, sid;
 pid = fork(); // Fork off the parent process
 if (pid < 0) {
   exit(EXIT_FAILURE);
 }
 if (pid > 0) {
   exit(EXIT_SUCCESS);
 }
 // Create a SID for child
 sid = setsid();
 if (sid < 0) {
   // FAIL
   exit(EXIT_FAILURE);
 }
 if ((chdir("/")) < 0) {
   // FAIL
   exit(EXIT_FAILURE);
 }
 close(STDIN_FILENO);
 close(STDOUT_FILENO);
 close(STDERR_FILENO);
 while (1) {
   // Some Tasks
   sleep(30);
 }
 exit(EXIT_SUCCESS);
}




int main()
{
 getchar();
 _daemon(0, 0);
 getchar();
 return 0;
}