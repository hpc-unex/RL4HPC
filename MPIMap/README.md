###########################
#########  GUIDE ##########
###########################


Clone repository: 

> git clone https://github.com/hpc-unex/colls.git



### HOW TO RECEIVE CHANGES:

Do this in colls/ folder. Check and update differences between github repository and your local repository.

> git fetch origin

> git diff master..origin/master

(*) Now it is IMPORTANT to ensure that different workers haven't change the same file. After check that with previous command, do the pull.

> git pull


### HOW TO DO CHANGES:

Do this in colls/ folder. After change files do the following.

IMPORTANT: first ensure that the version (HOW TO RECEIVE CHANGES) is not corrupted (*)

> git add .

> git commit -m "my_changes_comment"

> git push
