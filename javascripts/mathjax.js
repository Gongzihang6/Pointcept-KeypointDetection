window.MathJax = {
    tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true,
        // ⬇️ 核心配置：开启自动编号 ⬇️
        tags: "all",  // 选项: 'none' (默认), 'ams' (推荐), 'all' (所有公式都编号)
        tagSide: "right", // 编号在右侧
        tagIndent: "0.8em"
    },
    options: {
        ignoreHtmlClass: ".*|",
        processHtmlClass: "arithmatex"
    },
    // 新增：启动后监听 DOM 变化（针对某些动态加载场景）
    startup: {
        pageReady: () => {
            return MathJax.startup.defaultPageReady().then(() => {
                console.log("MathJax initial rendering done");
            });
        }
    }
};
