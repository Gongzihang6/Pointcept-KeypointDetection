---
hide:
  - navigation # 在首页隐藏右侧的本页目录（可选，让首页看起来更像封面）
---

<p class="theme-switcher-title">
  🎨 换个颜色，换个心情
</p>

<div class="color-picker-container">
  <button class="color-btn" data-color="red" style="background-color: #ef5350;">red</button>
  <button class="color-btn" data-color="pink" style="background-color: #ec407a;">pink</button>
  <button class="color-btn" data-color="purple" style="background-color: #ab47bc;">purple</button>
  <button class="color-btn" data-color="indigo" style="background-color: #5c6bc0;">indigo</button>
  <button class="color-btn" data-color="blue" style="background-color: #42a5f5;">blue</button>
  <button class="color-btn" data-color="cyan" style="background-color: #26c6da;">cyan</button>
  <button class="color-btn" data-color="teal" style="background-color: #26a69a;">teal</button>
  <button class="color-btn" data-color="green" style="background-color: #66bb6a;">green</button>
  <button class="color-btn" data-color="orange" style="background-color: #ffa726;">orange</button>
  <button class="color-btn" data-color="brown" style="background-color: #8d6e63;">brown</button>
  <button class="color-btn" data-color="grey" style="background-color: #bdbdbd;">grey</button>
  <button class="color-btn" data-color="black" style="background-color: #000000;">black</button>
</div>

<script>
  var buttons = document.querySelectorAll('.color-btn');
  var body = document.querySelector('body');
  buttons.forEach(function(btn) {
    btn.addEventListener('click', function() {
      var color = this.getAttribute('data-color');
      body.setAttribute('data-md-color-primary', color);
      localStorage.setItem('user-color-preference', color);
    });
  });
  var savedColor = localStorage.getItem('user-color-preference');
  if (savedColor) { body.setAttribute('data-md-color-primary', savedColor); }
</script>


# Pointcept-KeypointDetection


<script>
  var buttons = document.querySelectorAll('.color-btn');
  var body = document.querySelector('body');

  buttons.forEach(function(btn) {
    btn.addEventListener('click', function() {
      // 1. 获取按钮上存的颜色名
      var color = this.getAttribute('data-color');
      
      // 2. 修改 Material 主题的全局属性
      body.setAttribute('data-md-color-primary', color);
      
      // 3. (可选) 保存到本地缓存，刷新页面不丢失
      localStorage.setItem('user-color-preference', color);
    });
  });

  // 4. (可选) 页面加载时读取缓存
  var savedColor = localStorage.getItem('user-color-preference');
  if (savedColor) {
    body.setAttribute('data-md-color-primary', savedColor);
  }
</script>

> [!NOTE]
> **Pointcept-KeypointDetection** 是一个基于 Pointcept 框架的 3D 关键点检测项目。

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## ✨ 主要特性

* 🚀 **高性能**：基于 Pointcept 的高效实现。
* 📐 **精确**：针对 3D 点云的精确关键点定位。
* 🛠️ **易用**：模块化设计，易于扩展。

## 📦 快速预览

??? note "示例代码"
    ```python
    # 这是一个代码示例
    import pointcept
    print("Hello Pointcept!")
    ```



# 安装指南

## 环境要求

* Python >= 3.8
* PyTorch >= 1.10
* CUDA 可用

## 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/Gongzihang6/Pointcept-KeypointDetection.git
   ```
