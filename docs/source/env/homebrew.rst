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
