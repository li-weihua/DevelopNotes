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
