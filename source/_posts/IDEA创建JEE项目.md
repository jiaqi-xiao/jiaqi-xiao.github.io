---
layout: intellij
title: IDEA创建JEE项目
date: 2018-01-25 00:21:24
tags: JEE
---

### 环境准备：
1. Intellij IDEA 2017.2.5
2. JAVA jdk1.8.0_45
3. Tomcat 8.5
---
### 项目创建步骤
1. Create New Project
  ![初始界面](http://upload-images.jianshu.io/upload_images/8743973-8d9d89c37d902e7e.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
2. 创建JavaEE项目
  ![配置选项](http://upload-images.jianshu.io/upload_images/8743973-895fe7170bdcbca8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 项目名称
  ![项目名称](http://upload-images.jianshu.io/upload_images/8743973-ab7c0ef312db8292.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
3. 创建classes和lib文件
- File--> Project Structure --> Module --> Sources --> WEB-INF
  ![](http://upload-images.jianshu.io/upload_images/8743973-755cea0a1b43cf90.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 创建文件夹
  ![classes和lib](http://upload-images.jianshu.io/upload_images/8743973-6fb9b0eeecf02ad6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
4. Paths路径配置
- 选择Paths，选择Use Modules complie Output path，指定路径为上面创建的classes目录
  ![Paths](http://upload-images.jianshu.io/upload_images/8743973-1b13ca0bedff9022.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
5. 添加依赖路径
- 切换到 Dependencies  --> "+" -->JARs or directories... 
  ![Dependencies](http://upload-images.jianshu.io/upload_images/8743973-720e1884e94ffa56.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 选择创建的lib目录
  ![lib](http://upload-images.jianshu.io/upload_images/8743973-6507f4ac415c7047.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 选择Jar Directory
  ![Jar Directory](http://upload-images.jianshu.io/upload_images/8743973-23e04c1d0c430d4a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
  ![添加成功](http://upload-images.jianshu.io/upload_images/8743973-ee3cabcc9f50f13b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
6. 切换到 Artifacts选项卡，IDEA会为该项目自动创建一个名为“JavaEE_war exploded”的打包方式，表示 打包成war包，并且是文件展开性的，输出路径为当前项目下的 out 文件夹，保持默认即可。另外勾选下“Include in project build”，表示编译的时候就打包部署，勾选“Show content of elements”，表示显示详细的内容列表。
  ![Artifacts](http://upload-images.jianshu.io/upload_images/8743973-252025698c995aed.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
7. 配置Tomcat
- 点击Edit Configuration
  ![Edit Configuration](http://upload-images.jianshu.io/upload_images/8743973-c745744bedd77ea4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 添加配置
  ![Run/Debug Configuration](http://upload-images.jianshu.io/upload_images/8743973-f1d2a8c074861589.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 按下图设置
  ![](http://upload-images.jianshu.io/upload_images/8743973-6a54aa3e094cfe6e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 切换至Deployment
  ![Deployment](http://upload-images.jianshu.io/upload_images/8743973-dcf12e312de8cd60.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
8. 项目建立完成


参考：
http://blog.csdn.net/yhao2014/article/details/45740111
http://www.jianshu.com/p/455c7c11dfb2