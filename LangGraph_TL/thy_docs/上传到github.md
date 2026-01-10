# 删除 .git ，windows版本
Remove-Item -Recurse -Force .git




git init

git config --global user.email "....@qq.com"

git config --global user.name "hzqjgthy"

git add .


git status


git commit -m "20251130"

git remote add origin https://github.com/hzqjgthy/LangGraph_TL.git

git push -u origin main --force