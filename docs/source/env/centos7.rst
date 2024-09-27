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

