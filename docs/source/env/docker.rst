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
