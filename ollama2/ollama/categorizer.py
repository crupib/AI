import ollama

model = "llama3.2"

items = [
    "Passport",
    "Sunglasses",
    "Toothbrush",
    "Toothpaste",
    "Shampoo",
    "Conditioner",
    "Sunscreen",
    "First Aid Kit",
    "Phone Charger",
    "Power Bank",
    "Headphones",
    "Laptop",
    "Adapter",
    "Camera",
    "Notebook",
    "Pen",
    "Water Bottle",
    "Snacks",
    "Travel Pillow",
    "Blanket",
    "Umbrella",
    "Raincoat",
    "Hiking Boots",
    "Flip Flops",
    "Swimsuit",
    "Towel",
    "Jeans",
    "T-Shirts",
    "Sweater",
    "Jacket",
    "Socks",
    "Underwear",
    "Pajamas",
    "Watch",
    "Wallet",
    "Cash",
    "Credit Cards",
    "Travel Guide",
    "Books",
    "E-Reader",
    "Medications",
    "Hand Sanitizer",
    "Face Mask",
    "Tissues",
    "Sewing Kit",
    "Laundry Bag",
    "Reusable Shopping Bag"
]

prompt = f"""
You are an assistant that helps users organize items for a travel packing list.

Here is a list of items:
{items}

Please:
1. Categorize these items into groups such as Clothing, Toiletries, Food, Electronics, and Miscellaneous.
2. Sort the items alphabetically within each category.
"""

try:
    response = ollama.generate(
        model=model,
        prompt=prompt  # Contains both instructions and items
    )
    generated_text = response.get("response","")
    print("Travel Packing List:")
    print(generated_text)
except Exception as e:
    print("An error occurred:", str(e))