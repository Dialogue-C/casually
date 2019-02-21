# 搭建VPS

1. 在vultr上新建一个服务器（点击enabled ipv4 和第四个）

2. 摧毁旧服务器

3. 在命令行运行以下两句（若之前选的是Cent OS）

    wget --no-check-certificate  https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks.sh

    
    chmod +x shadowsocks.sh
    
    ./shadowsocks.sh 2>&1 | tee shadowsocks.log

4. 输入密码（默认：teddysun.com）、端口号（默认：9721）、加密方式（选项7）

5. 配置成功

    Congratulations, Shadowsocks-python server install completed!
    Your Server IP : 45.77.237.99
    Your Server Port : *****
    Your Password : **********
    Your Encryption Method: aes-256-cfb
    
    Welcome to visit:https://teddysun.com/342.html
    Enjoy it!

6. 切记restart！！