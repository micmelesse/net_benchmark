CURRENT_BRANCH=`git branch | grep \* | cut -d ' ' -f2`
git push origin $CURRENT_BRANCH -f