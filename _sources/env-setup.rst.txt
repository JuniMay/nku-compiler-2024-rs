环境配置
================================================

.. toctree::
   :hidden:
   :maxdepth: 4


.. attention::
    由于本实验指导手册面向 Rust 版本的编译器实现，所以许多工具、配置与 C++ 版本实验指导存在不同。

系统环境配置
---------------------

.. note::

    根据上机大作业的总体要求，你需要在 Linux 环境下完成实验。

.. note::

    建议将配置过程中所遇到的问题和解决方法写到实验报告中。

Windows 系统
^^^^^^^^^^^^^^^^^^^^^

对于 Windows 10/11 用户，可以通过 WSL2，VMware 或者 Virtual Box 等软件安装 Ubuntu 等
Linux 发行版进行实验（建议使用 WSL2）。WSL 对系统有一定的需求，可以
`在这里 <https://www.computerhope.com/issues/ch001879.htm#system-requirements>`_ 
找到相应说明。WSL 的安装说明可以在 `此处 <https://docs.microsoft.com/en-us/Windows/wsl/install-win10>`_
找到。

你可能会依次遇到 `4280 <https://github.com/microsoft/WSL/issues/4280>`_ 和
`5651 <https://github.com/microsoft/WSL/issues/5651>`_ 这两个错误。你可能需要升级 WSL
kernel，并借助官方工具将 Windows 升级至最新版本。

此外，VSCode 有插件可以帮助你在 Windows 平台上连接 WSL2 进行开发。

macOS 系统
^^^^^^^^^^^^^^^^^^^^^

macOS 用户可以使用 Docker 或者 `Orb <https://orbstack.dev>`_ 配置实验所需的 Linux 环境。Orb 类似于 Docker，但是提供了一个更加简单的配置方式。你可以通过 `官方文档 <https://docs.orbstack.dev>`_ 了解更多。

对于 Docker，你可以阅读这一份 `教程 <https://www.techdevpillar.com/blog/how-to-run-ubuntu-on-mac-with-docker/>`_ 启用 Ubuntu 容器 [#f1]_ 。
了解 Docker 与虚拟机的 `差异 <https://devopscon.io/blog/docker/docker-vs-virtual-machine-where-are-the-differences/>`_ 也会很有帮助。
如果你希望直接在 macOS 系统上进行实验，并且希望完成 RISC-V 后端的编译器，你可以将 `这个仓库 <https://github.com/riscv-software-src/homebrew-riscv>`_
添加到 Homebrew 中配置工具链。

Linux 系统
^^^^^^^^^^^^^^^^^^^^^

相信作为 Linux 用户，你已经基本熟悉了系统配置，可以直接在本地环境中进行实验。

.. [#f1] 使用 lastest ubuntu 即可。

Rust 环境配置
---------------------

首先安装 Rust 版本管理器 rustup 以及包管理器、构建工具 cargo，可以通过以下命令安装：

.. code-block:: bash

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

如果无法通过这一方法安装，可以先下载脚本，之后本地运行即可。如果在运行脚本时出现网络问题，可以修改
rustup 镜像，参考 `tuna 源的 rustup 帮助文档 <https://mirrors.tuna.tsinghua.edu.cn/help/rustup/>`_ 。

安装完成后你可能需要重新打开终端来让之前设置的环境变量生效。之后你可以通过以下命令验证 Rust 工具链的安装：

.. code-block:: bash

    rustc --version

你应该会看到一个类似于 ``rustc 1.80.0 (051478957 2024-07-21)`` 的输出。

对于 Cargo 这个包管理和构建工具的使用，可以参考 `The Cargo Book <https://doc.rust-lang.org/cargo/>`_ 。

对于 Rust 开发环境，推荐使用 VSCode + rust-analyzer 或者 JetBrains 的 RustRover IDE。

- VSCode: 你可能已经安装并在其他项目中使用过 VSCode 了，VSCode 可以很方便地通过安装插件来支持 Rust 开发。而且完全能够满足我们的实验需求。
- RustRover: RustRover 是 JetBrains 为 Rust 开发提供的 IDE，功能非常强大，提供了一些 Rust 语言的特性支持，但是比较庞大，而且需要额外安装。

.. note::
    如果你使用的是 Windows + WSL2，推荐使用 VSCode + rust-analyzer。这一组合已经经过助教测试，足够满足实验需求。不过 RustRover 也支持连接 WSL2，你可以自行尝试。

    如果你已经使用过 PyCharm，你可能会对 RustRover 更加熟悉，因为他们都是 JetBrains 的产品。

选择 VSCode 作为 IDE
^^^^^^^^^^^^^^^^^^^^^
你需要一个已经安装好的 VSCode，然后在 VSCode 中安装 rust-analyzer 扩展。可以参考 VSCode 官方提供的教程 `Getting Started with Rust <https://code.visualstudio.com/docs/languages/rust>`_ 。

.. note::
    如果你使用的是 Windows + WSL2，需要在主机 Windows 系统上安装 VSCode，然后在 WSL2 中安装 Rust 工具链（见后文）。之后在 WSL2 终端中通过 ``code .`` 命令调起 VSCode 进行开发。注意，你实际上需要将 rust-analyzer 扩展安装到 WSL2 的 VSCode Server 中。详情可以参考 `Developing in WSL <https://code.visualstudio.com/docs/remote/wsl>`_ 。

选择 RustRover 作为 IDE
^^^^^^^^^^^^^^^^^^^^^^^^^^
请于 `JetBrains 官网 <https://www.jetbrains.com/rust/>`_ 下载 RustRover，参考 `RustRover 官方文档 <https://www.jetbrains.com/help/rust/getting-started.html>`_ 进行安装。

.. note::
    有关 RustRover 连接 WSL2 的配置，请参考 `Connect to a project running on WSL2 <https://www.jetbrains.com/help/rust/remote-development-starting-page.html#run_in_wsl_ij>`_ 。

工具链配置
---------------------

我们的编译器构建最终要编译到 RISC-V 或者 ARM 汇编代码。对于大部分同学，你的电脑可能都是 x86_64 架构的，所以你需要安装交叉编译工具链。

.. note::

    对于 Windows + WSL2 用户，请在 WSL2 的 Linux 中安装工具链，而不是在 Windows 系统中安装。

.. note::

    对于 mac 用户，你可能已经在使用 ARM 架构的 CPU 了，所以如果你最终要完成 ARM 后端的编译器，大部分情况下并不需要额外的交叉编译工具链。

在安装具体的工具链之前，首先安装实验中可能用到的工具：

.. code-block:: bash

    sudo apt install build-essential
    sudo apt install llvm
    sudo apt install clang
    sudo apt install qemu-user
    sudo apt install gdb-multiarch

ARM 工具链
^^^^^^^^^^^^^^^^^^^^^

你可以使用 Linux 发行版的包管理工具安装对应的交叉编译工具链，例如，对于 Ubuntu，你可以使用以下命令安装 ARM 交叉编译工具链：

.. code-block:: bash

    sudo apt install gcc-arm-linux-gnueabi

.. note::

    实际上我们可以看到有 ``gcc-arm-linux-gnueabi`` 和 ``gcc-arm-linux-gnueabihf`` 两种格式，
    具体的差异推荐同学们自行查阅了解。

RISC-V 工具链
^^^^^^^^^^^^^^^^^^^^^

对于 RISC-V 交叉编译工具链，最简单的配置方式就是使用包管理器安装：

.. code-block:: bash

    sudo apt install gcc-riscv64-linux-gnu

.. note::

    你也可以尝试手动构建 `RISC-V GNU Compiler Toolchain <https://github.com/riscv-collab/riscv-gnu-toolchain>`_

.. tip::

    你可能还需要 ``objdump`` 等工具来查看编译后的二进制文件，你可以安装对应架构的 ``binutils`` 来获取这些工具。请善用 Google 和 ChatGPT 来获取更多信息。

.. important::

    如果你在环境配置的过程中遇到任何问题，请及时联系助教（但是在提问前请确认你已经阅读了 :doc:`intro` 中“如何解决问题”这一节）。
    同时我们建议你在实验报告中记录下配置过程中遇到的问题和解决方法。
