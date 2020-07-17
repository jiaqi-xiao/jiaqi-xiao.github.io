---
title: Transfer Hexo to New Device
date: 2020-07-17 12:08:24
tags: Programming
---

I established this personal blog with GitHub pages and Hexo on my Windows PC. Currently, I want to transfer from PC to MacOS. This article explains the whole process of transfering.

## Basic Thought

The basic thougth of the process is using GitHub branch to implement the cross platform synchronization. Keep the static webpage file in original master branch and create a new branch 'hexo' to backup all the soure code of the Hexo blog.

When these are all set, clone the hexo branch repo into new device and push the new updates to GitHub everytime renewing the blog contents. The static web page would be genrated and deployed in master branch by `hexo g -d`

Specific steps are shown as follows:

## Create new branch

-  `git clone` to clone your existing hexo blog repository.
- `git checkout -b hexo` create a new 'hexo' branch
- Delete all the files in this respo (especially in themes/) folder except '.git' folder.
- Copy the entire original hexo blog folder from your old device to current path.
- Delete all the old '.git' realted folders.
- `git add .` `git commit -m "init hexo blog src files"` `git push origin hexo:hexo` push to new branh on GitHub

## Install Hexo on new device

- Install node.js and git first. Use stable version node but not the latest version.
- `npm install hexo --save` install Hexo
- `npm install hexo-deployer-git --save` install git deployer plugin

## Deploy Blog

- add .gitignore in hexo branch
- add `.deploy_git/* public/*` in .gitigonre
- `hexo clean` clean the old public static webpages
- `hexo g -d` genreate and deploy new static webpages.

## Tips of Hexo CLI

```bash
hexo n "My blog" == hexo new "My blog" #New article
hexo g == hexo generate#Generate
hexo s == hexo server # start service preview
hexo d == hexo deploy#deploy
```



