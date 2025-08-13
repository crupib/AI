from openai import OpenAI
from dotenv import load_dotenv
import requests

load_dotenv()
client = OpenAI()

def get_website_html(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching the URL {url}: {e}")
        return ""
    
def extract_core_website_content(html: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"""
            You are a professional content extractor.
            Extract only the meaningful article or content body from the following HTML.
            Do not include headers, footers, ads, navigation links, or scripts.

            <html>
            {html}
            </html>

            Return plain text only.
        """
    )
    return response.output_text

def create_blog_post(content: str) -> str:
    response = client.responses.create(
        model="gpt-4o",
        input=f"""
            You are an expert blog writer.
            Write a well-structured blog post in markdown format based on the following text content:
            <source>
            {content}
            </source>

            Your blog post should include:
            - A clear and engaging title
            - A short introduction
            - 3–5 bullet points of key insights
            - A conclusion that wraps up the topic

            Use markdown formatting appropriately (e.g., # for title, ## for subheadings, * for bullet points).
        """
    )
    return response.output_text

def save_markdown_file(markdown_text: str, filename: str = "blog_post.md"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    print(f"\n✅ Blog post saved to: {filename}")

def main():
    website_url = input("🌐 Enter the website URL: ")
    print("🔄 Fetching website HTML...")
    html = get_website_html(website_url)

    if not html:
        print("❌ Unable to fetch HTML content. Exiting.")
        return
    
    print("🧠 Extracting core content...")
    core_content = extract_core_website_content(html)
    print("✔️ Core content extracted.")

    print("✍️ Generating blog post...")
    blog_markdown = create_blog_post(core_content)
    print("✔️ Blog post generated.")

    save_markdown_file(blog_markdown)  

if __name__ == "__main__":
    main()  