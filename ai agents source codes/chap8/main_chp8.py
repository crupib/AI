from openai import OpenAI
from dotenv import load_dotenv
import requests
import threading

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

def create_blog_post(content: str, iterations: int = 2) -> str:
    print("✍️ Generating initial blog post...")
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
    improved_post = response.output_text
    print("✅ Initial blog post generated.")
    print(improved_post)

    # Iteratively refine the blog post
    for i in range(1, iterations + 1):
        print(f"🔁 Refining blog post... (Iteration {i})")

        user_feedback = input("Enter feedback for this version (or press Enter to skip)") 

        refinement_prompt = f"""
                You are a professional editor and blog optimization expert.

                Please improve the following blog post by:
                - Enhancing clarity and tone
                - Making the structure more engaging
                - Improving formatting, flow, and transitions
        """

        if user_feedback.strip():
            refinement_prompt += f"""

            <user_feedback>
            {user_feedback}
            </user_feedback>
            """

        refinement_prompt += f"""
            <blog>
            {improved_post}
            </blog>

            Return the improved version in markdown format.
        """

        response = client.responses.create(
            model="gpt-4o",
            input=f"""
                You are a professional editor and blog optimization expert.

                Please improve the following blog post by:
                - Enhancing clarity and tone
                - Making the structure more engaging
                - Improving formatting, flow, and transitions

                <blog>
                {improved_post}
                </blog>

                Return the improved version in markdown format.
            """
        )
        improved_post = response.output_text
        print(f"✅ Refinement {i} completed.")
        print(improved_post)

        satisfied = input("Are you satisfied with this version? (yes to finish/no to continue)").strip().lower()

        if satisfied in {'yes':'y'}:
            print("Great! Final version confirmed")
            break	
    return improved_post

def save_markdown_file(markdown_text: str, filename: str = "blog_post.md"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    print(f"\n✅ Blog post saved to: {filename}")

def generate_thumbnail_image(article: str, filename: str = "thumbnail.png"):
    print("🖼️ Generating thumbnail image...")
    response = client.images.generate(
        model="dall-e-3",
        prompt=f"Generate a thumbnail for the following blog post: {article}",
        size="1024x1024",
        n=1
    )
    image_url = response.data[0].url

    image_data = requests.get(image_url).content
    with open(filename, "wb") as f:
        f.write(image_data)

    print(f"✅ Thumbnail image saved to: {filename}")

def generate_linkedin_post(blog_markdown: str, filename: str = "linkedin_post.txt"):
    print("💼 Generating LinkedIn post...")
    response = client.responses.create(
        model="gpt-4o",
        input=f"""
            You are a professional social media content writer.

            Write a LinkedIn post based on the following blog content:

            <blog>
            {blog_markdown}
            </blog>

            Your post should:
            - Be engaging and concise
            - Summarize the main point of the blog
            - Use a friendly yet professional tone
            - Include a call-to-action (e.g., “What do you think?”, “Share your thoughts below!”)
        """
    )
    linkedin_text = response.output_text
    with open(filename, "w", encoding="utf-8") as f:
        f.write(linkedin_text)
    print(f"✅ LinkedIn post saved to: {filename}")


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

    # Generate thumbnail and LinkedIn post in parallel
    thumbnail_thread = threading.Thread(target=generate_thumbnail_image, args=(blog_markdown,))
    linkedin_thread = threading.Thread(target=generate_linkedin_post, args=(blog_markdown,))

    thumbnail_thread.start()
    linkedin_thread.start()

    thumbnail_thread.join()
    linkedin_thread.join()      

if __name__ == "__main__":
    main()  