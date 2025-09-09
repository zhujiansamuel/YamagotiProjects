from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as p:
        # 启动浏览器（headless=False 表示可见模式）
        browser = p.chromium.launch(headless=False, slow_mo=500)
        page = browser.new_page()

        # 打开目标网页
        page.goto("https://www.mobile-ichiban.com/Prod/1/01")

        # 点击某个元素（示例：页面上的链接）
        # page.click("a")  # 用 CSS 选择器定位元素，这里点击第一个 <a> 标签

        # 保持浏览器打开，等待你手动关闭
        input("按回车键后关闭浏览器...")
        browser.close()

if __name__ == "__main__":
    main()
