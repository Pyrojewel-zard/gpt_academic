## 在 Windows 上配置 Git 秘钥并绑定 GitHub，再到 git pull 的全流程

本指南面向 Windows PowerShell 环境，涵盖从安装检查、SSH 秘钥生成与绑定、远程地址设置，到执行 `git pull` 的完整流程，并附常见问题排查。

### 0) 前置准备
- **安装 Git**：若未安装，请先安装 Git（含 Git Bash 与 Git Credential Manager）。
- **检查版本**：
```powershell
git --version
```

### 1) 配置全局用户信息
```powershell
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"

git config --global core.autocrlf true   # Windows 推荐
```

### 2) 生成 SSH 秘钥（推荐 Ed25519）
```powershell
# 将邮箱替换为你的 GitHub 邮箱
ssh-keygen -t ed25519 -C "你的邮箱@example.com"
# 连续回车使用默认路径：C:\Users\<你>\.ssh\id_ed25519
# 可设置一个安全的 passphrase（建议）
```

如遇到老旧环境不支持 Ed25519，可使用：
```powershell
ssh-keygen -t rsa -b 4096 -C "你的邮箱@example.com"
```

### 3) 启动 ssh-agent 并添加私钥
在 Windows PowerShell 中：
```powershell
# 启动 ssh-agent 服务（如已在运行会提示）
Get-Service ssh-agent | Set-Service -StartupType Automatic
Start-Service ssh-agent

# 将私钥加入 agent（路径按实际情况修改）
ssh-add $env:USERPROFILE\.ssh\id_ed25519
```

若提示无法找到文件，请确认秘钥路径是否正确，或将命令中的文件名改为 `id_rsa` 等。

### 4) 复制公钥内容
```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub | clip
```
上述命令会将公钥复制到剪贴板。

### 5) 将公钥添加到 GitHub 账户
1. 登录 GitHub，进入 Settings → SSH and GPG keys。
2. 点击 New SSH key，Title 随意（如 “My Windows PC”），Key 粘贴刚复制的公钥内容。
3. 保存。

参考文档（可选）：[Adding a new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)

### 6) 测试 SSH 连接
```powershell
ssh -T git@github.com
```
首次连接会提示加入 `known_hosts`，输入 `yes`。若成功，会看到类似：
```
Hi <username>! You've successfully authenticated, but GitHub does not provide shell access.
```

### 7) 将仓库远程地址设置为 SSH
有两种常见场景：

- **场景 A：新克隆仓库**
```powershell
git clone git@github.com:<USERNAME>/<REPO>.git
cd <REPO>
```

- **场景 B：已有仓库，当前远程是 HTTPS，需要切换为 SSH**
```powershell
cd <REPO>
git remote -v
git remote set-url origin git@github.com:<USERNAME>/<REPO>.git
git remote -v  # 确认已变更
```

若你使用的是组织或企业仓库，将 `<USERNAME>` 替换为组织名或按实际命名规则填写。

### 8) 首次拉取代码（git pull）
主分支可能是 `main` 或 `master`，以下以 `main` 为例：
```powershell
# 获取远程分支列表（可选）
git fetch origin

# 如果本地还没有 main 分支，先创建跟踪关系
git checkout -b main origin/main

# 日常拉取（使用 rebase 保持历史更干净）
git pull --rebase origin main
```

如果远程是 `master`：
```powershell
git checkout -b master origin/master
git pull --rebase origin master
```

### 9) 常见问题排查
- **Permission denied (publickey)**：
  - 检查公钥是否已添加到 GitHub。
  - 确认本机已 `ssh-add` 对应私钥：`ssh-add -l` 查看已加载的密钥。
  - 确认正在使用的远程地址是 SSH：`git remote -v` 中应以 `git@github.com:` 开头。

- **多个 SSH 密钥/账户共存**：使用 `~/.ssh/config` 指定 Host 别名与对应私钥：
```
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```
然后远程地址保持 `git@github.com:<USERNAME>/<REPO>.git` 即可。

- **ssh-agent 重启后丢失密钥**：
  - 确保 `ssh-agent` 服务为 Automatic 并已启动。
  - 如仍需每次登录自动添加，可编写登录脚本执行 `ssh-add`。

- **公司网络/代理导致连接问题**：
  - 先测试 `ssh -T git@github.com` 是否可达。
  - 必要时改用 HTTPS + Git Credential Manager 登录，或配置代理后再用 SSH。

### 10) 选用 HTTPS 的替代流程（可选）
若不方便使用 SSH，可使用 HTTPS：
```powershell
git clone https://github.com/<USERNAME>/<REPO>.git
cd <REPO>
git pull
```
首次推送或拉取需要登录时，Windows 的 Git Credential Manager 会引导你使用浏览器登录 GitHub（推荐使用 **Personal Access Token**）。

### 11) 快速命令清单（SSH 路线）
```powershell
# 基本配置
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"

# 生成并加载密钥
ssh-keygen -t ed25519 -C "你的邮箱@example.com"
Get-Service ssh-agent | Set-Service -StartupType Automatic
Start-Service ssh-agent
ssh-add $env:USERPROFILE\.ssh\id_ed25519

# 添加公钥到 GitHub 后，测试连接
ssh -T git@github.com

# 设置/切换远程为 SSH 并拉取
git remote set-url origin git@github.com:<USERNAME>/<REPO>.git
git pull --rebase origin main
```

---
如需团队协作工作流（fork 同步、rebase、PR 等），可扩展至标准化 Git 工作流：
- 本地保持 `main` 干净，功能在分支上开发；
- 定期 `git fetch` + `git rebase origin/main` 保持分支最新；
- 提交信息使用 Conventional Commits（例如：`feat: ...`、`fix: ...`）。


