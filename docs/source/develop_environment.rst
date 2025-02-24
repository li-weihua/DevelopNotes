**********************
Develop Environment
**********************

Centos7
=========

How to install higher version gcc on centos7 docker?
-------------------------------------------------------

Ref: https://stackoverflow.com/questions/67090507/how-to-install-gcc-g-9-on-centos-7-docker-centos7

.. code-block:: dockerfile

    FROM centos:7 AS env

    RUN yum update -y && \
        yum install -y centos-release-scl && \
        yum install -y devtoolset-9 && \
        echo "source /opt/rh/devtoolset-9/enable" >> /etc/bashrc


CMAKE
=======

cudatoolkit version and supported gpus
----------------------------------------


How to get the version of library found by CMake?
--------------------------------------------------

Ref: https://stackoverflow.com/questions/34138886/how-to-know-version-of-library-found-by-cmake

``<package>_VERSION``


How to build x86 or x64 on Windows from command line with CMAKE?
--------------------------------------------------------------------

Ref: https://stackoverflow.com/questions/28350214/how-to-build-x86-and-or-x64-on-windows-from-command-line-with-cmake

.. code-block:: bash

    cmake -G "Visual Studio 17 2022" -A Win32 -S .. -B "build32"
    cmake -G "Visual Studio 17 2022" -A x64 -S .. -B "build64"
    cmake --build build32 --config Release
    cmake --build build64 --config Release


For simplicity, only build 64bit version:

.. code-block:: bash

    cmake .. -A x64
    cmake --build . --config Release


ld: warning: missing .note.GNU-stack section implies executable stack
----------------------------------------------------------------------------

This is a new warning message introduced by GNU bin-utils version 2.39.

出现“ld missing .note.GNU-stack section implies executable stack”的错误提示，
通常是因为在链接过程中，某些对象文件缺少.note.GNU-stack节，导致链接器默认认为可能需要一个可执行的栈。
这在现代操作系统中是不推荐的做法，因为可执行的栈可能会导致安全问题，如缓冲区溢出攻击。

需要在汇编码中加入：

``.section    .note.GNU-stack,"",@progbits``

或者编译选项：


.. code-block:: cmake

    if(UNIX)
        set_property(TARGET target APPEND_STRING PROPERTY LINK_FLAGS "-z noexecstack")
    endif()



Conda
=======

conda命令补全
--------------

参考： https://github.com/tartansandal/conda-bash-completion

采用方法2直接把 https://github.com/tartansandal/conda-bash-completion/blob/master/conda 下载到bash补全目录

Docker
========

Manage Docker as a non-root user
------------------------------------

reference: https://docs.docker.com/engine/install/linux-postinstall/

1. If ``docker`` group does not exit:

.. code-block:: bash

    sudo groupadd docker

2. Add your user to the ``docker`` group:

.. code-block:: bash

    sudo usermod -aG docker $USER

3. To activate the changes to groups:

.. code-block:: bash

    newgrp docker


manylinux
===========

centos7, glibc version 2.17

git
=====

在 HTTPS 端口使用 SSH
-----------------------
参考：https://docs.github.com/zh/authentication/troubleshooting-ssh/using-ssh-over-the-https-port

.. code-block::

    Host github.com
        Hostname ssh.github.com
        Port 443
        User git


Homebrew
==========

由于tencent的homebrew源有问题，所以暂用科大源替代。

科大源安装 Homebrew / Linuxbrew
---------------------------------

首先在命令行运行如下几条命令设置环境变量：

.. code-block:: bash

    export HOMEBREW_BREW_GIT_REMOTE="https://mirrors.ustc.edu.cn/brew.git"
    export HOMEBREW_BOTTLE_DOMAIN="https://mirrors.ustc.edu.cn/homebrew-bottles"
    export HOMEBREW_API_DOMAIN="https://mirrors.ustc.edu.cn/homebrew-bottles/api"

之后在命令行运行 Homebrew 安装脚本：

.. code-block:: bash

    /bin/bash -c "$(curl -fsSL https://github.com/Homebrew/install/raw/HEAD/install.sh)"


.. note::
    若用户设置了环境变量 ``HOMEBREW_BREW_GIT_REMOTE``，则每次运行 ``brew update`` 时将会自动设置远程。
    推荐用户将环境变量 ``HOMEBREW_BREW_GIT_REMOTE`` 加入 shell 的 profile 设置中。

.. code-block:: bash

        # 对于 bash 用户
        echo 'export HOMEBREW_BREW_GIT_REMOTE="https://mirrors.ustc.edu.cn/brew.git"' >> ~/.bash_profile

        # 对于 zsh 用户
        echo 'export HOMEBREW_BREW_GIT_REMOTE="https://mirrors.ustc.edu.cn/brew.git"' >> ~/.zshrc

重置为官方地址：

.. code-block:: bash

    unset HOMEBREW_BREW_GIT_REMOTE
    git -C "$(brew --repo)" remote set-url origin https://github.com/Homebrew/brew

.. note::
    重置回默认远程后，用户应该删除 shell 的 profile 设置中的环境变量 ``HOMEBREW_BREW_GIT_REMOTE`` 以免运行 ``brew update`` 时远程再次被更换。

    若之前使用的 ``git config url.<URL>.insteadOf URL`` 的方式设置的镜像，请手动删除 ``config`` 文件（一般为 ``~/.gitconfig`` 或仓库目录下的 ``.git/config``）中的对应字段。


Linux Homebrew参考设置
--------------------------

.. code-block:: bash

    export HOMEBREW_BREW_GIT_REMOTE="https://mirrors.ustc.edu.cn/brew.git"
    export HOMEBREW_BOTTLE_DOMAIN="https://mirrors.ustc.edu.cn/homebrew-bottles"
    export HOMEBREW_API_DOMAIN="https://mirrors.ustc.edu.cn/homebrew-bottles/api"

    eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

    if type brew &>/dev/null
    then
        HOMEBREW_PREFIX="$(brew --prefix)"
        if [[ -r "${HOMEBREW_PREFIX}/etc/profile.d/bash_completion.sh" ]]
        then
            source "${HOMEBREW_PREFIX}/etc/profile.d/bash_completion.sh"
        else
            for COMPLETION in "${HOMEBREW_PREFIX}/etc/bash_completion.d/"*
            do
                [[ -r "${COMPLETION}" ]] && source "${COMPLETION}"
            done
        fi
    fi


通过brew安装的vim启动时出现错误
--------------------------------

解决方法：

.. code-block:: bash

    brew reinstall --build-from-source vim


brew相关链接
-------------

:科大文档: https://mirrors.ustc.edu.cn/help/brew.git.html
:官方主页: http://brew.sh/
:brew 文档: http://docs.brew.sh/
