### **Business Use Case: Personalized Product Recommender for E-commerce**

#### **Scenario:**
You are working for a large e-commerce company similar to Amazon or Walmart, and your goal is to create a personalized product recommendation system for the platform. The e-commerce site offers thousands of products across various categories like electronics, home appliances, fashion, and groceries. Customers often browse and purchase products but struggle to discover new items that match their preferences or needs.

A powerful recommendation system can help increase sales by suggesting products that customers are likely to purchase based on their past behavior or the attributes of products they have shown interest in. The Content-Based Recommender you will build uses product metadata, such as product descriptions, categories, brand, and price, to find similar products. For example, if a customer viewed a high-end laptop, your system should recommend other laptops with similar specifications or other related electronics.

The content-based approach is particularly useful for new users who have not made many purchases but have browsed a few products, as it relies on the attributes of the products themselves rather than user purchase history.

#### **Dataset Overview**
The dataset you will use includes information about various products, including:

1. **Product ID**: A unique identifier for the product.
2. **Title**: The name of the product.
3. **Description**: A brief description of the product, including specifications and key features.
4. **Category**: The category or categories associated with the product (e.g., Electronics, Home Appliances).
5. **Brand**: The brand of the product.
6. **Price**: The price of the product.
7. **Rating**: The average rating given by users.
8. **Number of Ratings**: The total number of ratings the product has received.

#### **Use Case:**
**Personalized Product Recommendations for Online Shoppers**

Suppose a customer recently viewed a high-end camera by Sony with a 4.5-star rating. Your system should recommend other cameras with similar specifications (e.g., 4K resolution, interchangeable lenses) or products in related categories, such as camera accessories, memory cards, and camera bags. Additionally, if the customer has browsed high-end electronics, the system should prioritize recommending other high-quality, premium electronics rather than budget options.

Your system should focus on providing relevant, content-based recommendations to help customers discover new products they might not have considered but are likely to buy.

---

### **Lab: Building a Content-Based Product Recommender System**

### **Step 1: Setup and Load the Dataset**

1. **Import necessary libraries**:

   ```python
   import pandas as pd
   import numpy as np
   ```

2. **Load the product dataset**:

   ```python
   df_products = pd.read_csv('/mnt/data/product_dataset.csv')  # Replace with your dataset path
   ```

3. **Explore the dataset**:

   ```python
   print(df_products.head())
   ```

   Take a look at the first few rows to understand the data structure and columns.

### **Step 2: Data Preprocessing**

1. **Handle missing values**:

   Check for missing values and either drop or fill them as needed:

   ```python
   df_products.dropna(subset=['category', 'description', 'brand', 'price', 'rating'], inplace=True)
   ```

2. **Convert categorical data (e.g., categories, brand) into a format suitable for analysis**:

   The `category` and `brand` columns may need to be tokenized or converted into a list of keywords:

   ```python
   df_products['category_list'] = df_products['category'].apply(lambda x: x.split(', '))
   ```

3. **Create a new column combining relevant content features**:

   Combine `description`, `category`, and `brand` into a single `content` column for better feature representation:

   ```python
   df_products['content'] = df_products['description'] + ' ' + df_products['category_list'].apply(lambda x: ' '.join(x)) + ' ' + df_products['brand']
   ```

### **Step 3: Build the Recommender System**

1. **Convert text data into numerical format**:

   Use `TfidfVectorizer` to convert the `content` column into numerical vectors:

   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer

   tfidf_vectorizer = TfidfVectorizer(stop_words='english')
   tfidf_matrix_products = tfidf_vectorizer.fit_transform(df_products['content'])
   ```

2. **Calculate similarity**:

   Use cosine similarity to find the similarity between products:

   ```python
   from sklearn.metrics.pairwise import linear_kernel

   cosine_sim_products = linear_kernel(tfidf_matrix_products, tfidf_matrix_products)
   ```

3. **Create a function to get product recommendations**:

   - Create a reverse mapping of product titles and their indices:

     ```python
     indices_products = pd.Series(df_products.index, index=df_products['title']).drop_duplicates()
     ```

   - Define the recommendation function:

     ```python
     def get_product_recommendations(title, cosine_sim=cosine_sim_products):
         # Get the index of the product that matches the title
         idx = indices_products[title]

         # Get the pairwise similarity scores of all products with that product
         sim_scores = list(enumerate(cosine_sim[idx]))

         # Sort the products based on the similarity scores
         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

         # Get the scores of the 10 most similar products
         sim_scores = sim_scores[1:11]

         # Get the product indices
         product_indices = [i[0] for i in sim_scores]

         # Return the top 10 most similar products
         return df_products[['title', 'price', 'brand', 'category']].iloc[product_indices]
     ```

### **Step 4: Test the Recommender System**

Test the recommender system with a sample product title:

```python
print("Recommendations for 'Sony Alpha a7 III':")
print(get_product_recommendations('Sony Alpha a7 III'))
```

### **Step 5: Analyze and Discuss Results**

1. Look at the recommendations and see if they align with user expectations.
2. Discuss how changing the weight of certain features (e.g., placing more emphasis on price or brand) could affect the recommendations.

### **Step 6: Advanced Enhancements (Optional)**

1. **Incorporate Price Sensitivity**: Adjust recommendations based on price ranges similar to the viewed product.
2. **Use Brand Awareness**: Increase similarity scores for products of the same brand.
3. **Hybrid Recommendation**: Combine this content-based recommender with collaborative filtering for a more comprehensive recommendation system.

### **Step 7: Save and Share**

Save your notebook and share it with your peers or stakeholders for review and feedback.

### **Conclusion**

In this lab, you built a content-based recommender system that provides personalized product recommendations based on product metadata. You learned how to preprocess text data, calculate similarities, and make recommendations using features like product description, category, brand, and price. This type of recommender system can be used to enhance user experience on e-commerce platforms by providing highly relevant product suggestions.
