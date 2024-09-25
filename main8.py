import streamlit as st
import tensorflow as tf
import numpy as np
from googletrans import Translator

# Language dictionary for translation
language_dict = {
    'English': 'en',
    'Tamil': 'ta',
    'Hindi': 'hi',
    'Telugu': 'te',
    'Malayalam': 'ml',
    'Kannada': 'kn',
    'Gujarati': 'gu',
    'Marathi': 'mr'
}

# Initialize Translator
translator = Translator()

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Water footprint data for fruits and vegetables
water_footprint_data = {
    'banana': '790 liters/kg',
    'apple': '820 liters/kg',
    'pear': '920 liters/kg',
    'grapes': '610 liters/kg',
    'orange': '560 liters/kg',
    'kiwi': '520 liters/kg',
    'watermelon': '50 liters/kg',
    'pomegranate': '550 liters/kg',
    'pineapple': '450 liters/kg',
    'mango': '900 liters/kg',
    'cucumber': '240 liters/kg',
    'carrot': '130 liters/kg',
    'capsicum': '110 liters/kg',
    'onion': '410 liters/kg',
    'potato': '290 liters/kg',
    'lemon': '70 liters/kg',
    'tomato': '180 liters/kg',
    'raddish': '80 liters/kg',
    'beetroot': '90 liters/kg',
    'cabbage': '120 liters/kg',
    'lettuce': '130 liters/kg',
    'spinach': '240 liters/kg',
    'soy bean': '2,145 liters/kg',
    'cauliflower': '140 liters/kg',
    'bell pepper': '100 liters/kg',
    'chilli pepper': '120 liters/kg',
    'turnip': '70 liters/kg',
    'corn': '120 liters/kg',
    'sweetcorn': '100 liters/kg',
    'sweet potato': '320 liters/kg',
    'paprika': '95 liters/kg',
    'jalepeño': '140 liters/kg',
    'ginger': '600 liters/kg',
    'garlic': '710 liters/kg',
    'peas': '450 liters/kg',
    'eggplant': '200 liters/kg'
}

# Health benefits for fruits and vegetables
health_benefits_data = {
    'banana': 'Rich in potassium and good for heart health.',
    'apple': 'High in fiber and antioxidants, supports heart health and digestion.',
    'pear': 'Rich in fiber and vitamin C, aids in digestion and boosts immunity.',
    'grapes': 'Contains antioxidants and supports heart health and hydration.',
    'orange': 'High in vitamin C, boosts immunity and promotes skin health.',
    'kiwi': 'Rich in vitamin C and fiber, supports digestion and immune function.',
    'watermelon': 'Hydrating and low in calories, rich in vitamins A and C.',
    'pomegranate': 'High in antioxidants, supports heart health and has anti-inflammatory properties.',
    'pineapple': 'Contains bromelain, aids digestion and supports immune health.',
    'mango': 'Rich in vitamins A and C, supports eye health and digestion.',
    'cucumber': 'Hydrating and low in calories, supports skin health.',
    'carrot': 'High in beta-carotene, supports eye health and immune function.',
    'capsicum': 'Rich in vitamins A and C, supports immune health and skin health.',
    'onion': 'Contains antioxidants, may support heart health and reduce inflammation.',
    'potato': 'Rich in potassium and vitamins, provides energy and supports digestive health.',
    'lemon': 'High in vitamin C, aids digestion and boosts immunity.',
    'tomato': 'Rich in lycopene, supports heart health and skin health.',
    'raddish': 'Low in calories, aids digestion and supports detoxification.',
    'beetroot': 'Rich in nitrates, supports blood flow and lowers blood pressure.',
    'cabbage': 'High in fiber and vitamins, supports digestion and immune health.',
    'lettuce': 'Low in calories, high in fiber, supports weight management.',
    'spinach': 'Rich in iron and vitamins, supports muscle and bone health.',
    'soy bean': 'High in protein and healthy fats, supports heart health and muscle growth.',
    'cauliflower': 'Low in calories and high in fiber, supports weight loss and digestion.',
    'bell pepper': 'Rich in vitamins A and C, supports immune health and skin health.',
    'chilli pepper': 'Contains capsaicin, may boost metabolism and support weight loss.',
    'turnip': 'Low in calories, high in vitamins and minerals, supports digestion.',
    'corn': 'Good source of carbohydrates and fiber, provides energy.',
    'sweetcorn': 'Rich in fiber and antioxidants, supports digestion and heart health.',
    'sweet potato': 'High in beta-carotene and fiber, supports eye health and digestion.',
    'paprika': 'Contains antioxidants, may support metabolism and reduce inflammation.',
    'jalepeño': 'Contains capsaicin, may help boost metabolism and support heart health.',
    'ginger': 'Anti-inflammatory properties, aids digestion and may reduce nausea.',
    'garlic': 'Rich in antioxidants, may support heart health and immune function.',
    'peas': 'High in protein and fiber, supports digestion and muscle health.',
    'eggplant': 'Rich in fiber and antioxidants, supports heart health and digestion.'
}

# Alternative, less water footprint items
alternative_items = {
    'banana': ['apple', 'grapes', 'watermelon'],
    'apple': ['pear', 'grapes', 'orange'],
    'pear': ['apple', 'grapes', 'orange'],
    'mango': ['pineapple', 'papaya', 'peach'],
    'pomegranate': ['strawberry', 'blueberry', 'raspberry'],
    'pineapple': ['kiwi', 'orange', 'apple'],
    'potato': ['sweet potato', 'yam', 'carrot'],
    'tomato': ['cucumber', 'bell pepper', 'zucchini'],
    'spinach': ['lettuce', 'kale', 'chard'],
    'soy bean': ['chickpeas', 'lentils', 'peas'],
    'lemon': ['lime', 'orange', 'grapefruit'],
    'ginger': ['turmeric', 'galangal', 'garlic'],
    'garlic': ['onion', 'shallots', 'chives'],
    'cauliflower': ['broccoli', 'brussels sprouts', 'cabbage'],
    'carrot': ['beetroot', 'radish', 'parsnip'],
    'onion': ['shallots', 'scallions', 'leeks'],
    'cucumber': ['zucchini', 'celery', 'bell pepper'],
    'eggplant': ['zucchini', 'bell pepper', 'mushroom'],
    'watermelon': ['cantaloupe', 'honeydew', 'grapefruit'],
    'kiwi': ['strawberry', 'grapes', 'peach'],
    'capsicum': ['bell pepper', 'jalapeño', 'habanero'],
    'jalapeño': ['serrano pepper', 'habanero', 'banana pepper'],
    'beetroot': ['carrot', 'radish', 'turnip'],
    'raddish': ['turnip', 'beetroot', 'carrot'],
    'sweet potato': ['yam', 'butternut squash', 'pumpkin'],
    'paprika': ['cayenne pepper', 'chili powder', 'red bell pepper']
}

# Final product water footprint data
final_product_footprint_data = {
'chappathi': '450 liters per piece',
'tomato chutney': '179 liters per batch',
'dosa': '400 liters per dosa',
'sambar rice': '1200 liters per serving',
'idli ': '300 liters per idli',
'paneer tikka': '1400 liters per serving',
'banana smoothie':'1000 liters per liter',
'apple pie':'1500 liters per kilogram',
'pear jam':'900 liters per kilogram',
'grape juice':'610 liters per liter',
'orange juice':'1000 liters per liter',
'kiwi sorbet':'500 liters per liter',
'watermelon juice':'250 liters per liter',
'pomegranate juice':'850 liters per liter',
'pineapple juice':'260 liters per liter',
'mango lassi':'2000 liters per liter',
'cucumber salad':'250 liters per kilogram',
'carrot soup':'150 liters per liter',
'stuffed bell peppers':'500 liters per kilogram',
'onion rings':'300 liters per kilogram',
'mashed potatoes':'300 liters per kilogram',
'lemonade':'1000 liters per liter',
'tomato sauce':'330 liters per kilogram',
'radish pickles':'150 liters per kilogram',
'beetroot salad':'300 liters per kilogram',
'cabbage stew':'250 liters per kilogram',
'lettuce wraps':'250 liters per kilogram',
'spinach dip':'400 liters per kilogram',
'soy milk':'2500 liters per liter',
'cauliflower rice':'300 liters per kilogram',
'grilled bell pepper':'250 liters per kilogram',
'chilli sauce':'300 liters per liter',
'turnip soup':'300 liters per liter',
'corn tortilla':'600 liters per kilogram',
'sweetcorn fritters':'500 liters per kilogram',
'sweet potato fries':'350 liters per kilogram',
'paprika seasoning':'300 liters per kilogram',
'jalapeño poppers':'350 liters per kilogram',
'ginger tea':'600 liters per liter',
'garlic bread':'1200 liters per kilogram',
'pea soup':'2500 liters per liter',
'eggplant parmesan':'400 liters per kilogram',
'banana bread':'2000 liters per kilogram',
'apple crumble':'1800 liters per kilogram',
'pear tart':'1500 liters per kilogram',
'grape jelly':'700 liters per kilogram',
'orange marmalade':'1200 liters per kilogram',
'kiwi jam':'600 liters per kilogram',
'watermelon sorbet':'300 liters per liter',
'pomegranate salad dressing':'900 liters per liter',
'pineapple upside-down cake':'1800 liters per kilogram',
'mango chutney':'1700 liters per kilogram',
'cucumber pickles':'300 liters per kilogram',
'carrot cake':'2000 liters per kilogram',
'stuffed onion':'350 liters per kilogram',
'potato chips':'1000 liters per kilogram'
}

# Health benefits for final products
health_benefits_final_product = {
    'chappathi': 'Provides energy and fiber, helps in digestion.',
    'tomato chutney': 'Rich in antioxidants, supports heart health, and aids in digestion.',
    'dosa': 'High in carbohydrates and protein, aids in weight management.',
    'sambar rice': 'Rich in proteins and fiber, helps regulate blood sugar levels.',
    'idli': 'Low in calories, good source of probiotics, supports gut health.',
    'paneer tikka': 'High in protein and calcium, promotes muscle health.',
    'banana smoothie': 'Rich in potassium and vitamins, provides energy and aids digestion.',
    'apple pie': 'Contains antioxidants and fiber, supports heart health.',
    'pear jam': 'Rich in fiber and vitamins, supports digestion.',
    'grape juice': 'High in antioxidants, supports heart health and hydration.',
    'orange juice': 'Rich in vitamin C, boosts immunity and hydration.',
    'kiwi sorbet': 'High in vitamin C, aids digestion and skin health.',
    'watermelon juice': 'Hydrating, low in calories, and rich in vitamins.',
    'pomegranate juice': 'Rich in antioxidants, supports heart health and anti-inflammatory properties.',
    'pineapple juice': 'Contains bromelain, aids digestion and reduces inflammation.',
    'mango lassi': 'Rich in probiotics, provides hydration and nutrients.',
    'cucumber salad': 'Hydrating and low in calories, supports skin health.',
    'carrot soup': 'High in beta-carotene, supports eye health and immune function.',
    'stuffed bell peppers': 'High in vitamins and antioxidants, promotes healthy skin.',
    'onion rings': 'Contains antioxidants, supports heart health in moderation.',
    'mashed potatoes': 'Rich in potassium and carbohydrates, provides energy.',
    'lemonade': 'Rich in vitamin C, aids in digestion and hydration.',
    'tomato sauce': 'Rich in lycopene, supports heart health and skin health.',
    'radish pickles': 'Low in calories, aids digestion and detoxification.',
    'beetroot salad': 'Rich in nitrates, supports blood flow and lowers blood pressure.',
    'cabbage stew': 'High in fiber and vitamins, supports digestion and immune health.',
    'lettuce wraps': 'Low in calories, high in fiber, supports weight management.',
    'spinach dip': 'Rich in iron and vitamins, supports muscle and bone health.',
    'soy milk': 'High in protein, supports heart health and bone density.',
    'cauliflower rice': 'Low in calories, high in fiber, promotes weight loss.',
    'grilled bell pepper': 'Rich in vitamins A and C, supports eye health.',
    'chilli sauce': 'Contains capsaicin, may boost metabolism and support weight loss.',
    'turnip soup': 'Low in calories, high in vitamins and minerals.',
    'corn tortilla': 'Good source of carbohydrates and fiber, provides energy.',
    'sweetcorn fritters': 'Rich in fiber, supports digestion and energy.',
    'sweet potato fries': 'High in beta-carotene, supports eye health.',
    'paprika seasoning': 'Contains antioxidants, may support metabolism.',
    'jalapeño poppers': 'Contains capsaicin, may help boost metabolism.',
    'ginger tea': 'Anti-inflammatory properties, aids digestion and may reduce nausea.',
    'garlic bread': 'Contains antioxidants, may support heart health.',
    'pea soup': 'High in protein and fiber, supports digestion and muscle health.',
    'eggplant parmesan': 'Rich in fiber and antioxidants, supports heart health.',
    'banana bread': 'Provides energy and fiber, supports digestive health.',
    'apple crumble': 'Contains antioxidants and fiber, supports heart health.',
    'pear tart': 'Rich in fiber and vitamins, aids digestion.',
    'grape jelly': 'Contains antioxidants, supports heart health.',
    'orange marmalade': 'Rich in vitamin C, boosts immunity and supports skin health.',
    'kiwi jam': 'High in vitamin C, aids digestion and boosts immunity.',
    'watermelon sorbet': 'Hydrating and low in calories, rich in vitamins.',
    'pomegranate salad dressing': 'Rich in antioxidants, supports heart health.',
    'pineapple upside-down cake': 'Contains bromelain, aids digestion and reduces inflammation.',
    'mango chutney': 'Rich in vitamins and antioxidants, supports immune health.',
    'cucumber pickles': 'Low in calories, aids digestion and hydration.',
    'carrot cake': 'Contains fiber and vitamins, supports eye health.',
    'stuffed onion': 'Rich in antioxidants, supports heart health.',
    'potato chips': 'Provides energy, but should be consumed in moderation due to high fat content.',

}

# Alternative final products
alternative_final_products = {
    'chappathi': ['roti (300 liters per piece)', 'poori (380 liters per piece)', 'phulka (350 liters per piece)'],
    'tomato chutney': ['mint chutney (120 liters per batch)', 'coriander chutney (100 liters per batch)', 'coconut chutney (140 liters per batch)'],
    'dosa': ['uttapam (350 liters per piece)', 'appam (320 liters per piece)', 'pancake (250 liters per piece)'],
    'sambar rice': ['curd rice (800 liters per serving)', 'lemon rice (900 liters per serving)', 'coconut rice (700 liters per serving)'],
    'idli': ['rava idli (250 liters per idli)', 'poha (200 liters per serving)', 'bread idli (280 liters per idli)'],
    'paneer tikka': ['grilled tofu (1000 liters per serving)', 'mushroom tikka (900 liters per serving)', 'vegetable skewers (850 liters per serving)'],
    'apple pie': ['pear tart (1300 liters per kilogram)', 'banana muffin (1000 liters per kilogram)', 'carrot cake (900 liters per kilogram)'],
    'pear jam': ['apple butter (700 liters per kilogram)', 'apricot jam (600 liters per kilogram)', 'plum jam (500 liters per kilogram)'],
    'grape juice': ['apple juice (400 liters per liter)', 'cranberry juice (500 liters per liter)', 'watermelon juice (250 liters per liter)'],
    'orange juice': ['lemonade (500 liters per liter)', 'pineapple juice (260 liters per liter)', 'grape juice (610 liters per liter)'],
    'kiwi sorbet': ['lemon sorbet (400 liters per liter)', 'watermelon sorbet (300 liters per liter)', 'orange sorbet (450 liters per liter)'],
    'watermelon juice': ['cucumber water (150 liters per liter)', 'coconut water (200 liters per liter)', 'cantaloupe juice (220 liters per liter)'],
    'pomegranate juice': ['grape juice (610 liters per liter)', 'cherry juice (700 liters per liter)', 'orange juice (1000 liters per liter)'],
    'pineapple juice': ['cucumber juice (150 liters per liter)', 'carrot juice (200 liters per liter)', 'watermelon juice (250 liters per liter)'],
    'mango lassi': ['buttermilk (1000 liters per liter)', 'strawberry lassi (1500 liters per liter)', 'coconut lassi (1800 liters per liter)'],
    'cucumber salad': ['tomato salad (200 liters per kilogram)', 'lettuce salad (180 liters per kilogram)', 'cabbage salad (200 liters per kilogram)'],
    'carrot soup': ['tomato soup (100 liters per liter)', 'zucchini soup (120 liters per liter)', 'pumpkin soup (130 liters per liter)'],
    'stuffed bell peppers': ['stuffed tomatoes (400 liters per kilogram)', 'stuffed zucchini (350 liters per kilogram)', 'grilled bell pepper (250 liters per kilogram)'],
    'onion rings': ['zucchini fries (200 liters per kilogram)', 'sweet potato fries (350 liters per kilogram)', 'carrot fries (250 liters per kilogram)'],
    'mashed potatoes': ['mashed cauliflower (200 liters per kilogram)', 'mashed sweet potatoes (250 liters per kilogram)', 'mashed turnips (230 liters per kilogram)'],
    'lemonade': ['cucumber water (150 liters per liter)', 'iced tea (400 liters per liter)', 'pineapple juice (260 liters per liter)'],
    'tomato sauce': ['marinara sauce (300 liters per kilogram)', 'red pepper sauce (280 liters per kilogram)', 'mushroom sauce (250 liters per kilogram)'],
    'radish pickles': ['cucumber pickles (200 liters per kilogram)', 'carrot pickles (150 liters per kilogram)', 'beetroot pickles (170 liters per kilogram)'],
    'beetroot salad': ['carrot salad (250 liters per kilogram)', 'cabbage salad (200 liters per kilogram)', 'radish salad (180 liters per kilogram)'],
    'cabbage stew': ['lentil stew (200 liters per kilogram)', 'spinach stew (230 liters per kilogram)', 'kale stew (240 liters per kilogram)'],
    'lettuce wraps': ['cabbage wraps (200 liters per kilogram)', 'collard greens wraps (230 liters per kilogram)', 'swiss chard wraps (220 liters per kilogram)'],
    'spinach dip': ['kale dip (350 liters per kilogram)', 'artichoke dip (320 liters per kilogram)', 'eggplant dip (300 liters per kilogram)'],
    'soy milk': ['almond milk (1200 liters per liter)', 'rice milk (1400 liters per liter)', 'oat milk (900 liters per liter)'],
    'cauliflower rice': ['broccoli rice (250 liters per kilogram)', 'zucchini rice (220 liters per kilogram)', 'cabbage rice (200 liters per kilogram)'],
    'grilled bell pepper': ['grilled zucchini (200 liters per kilogram)', 'grilled eggplant (250 liters per kilogram)', 'grilled mushrooms (220 liters per kilogram)'],
    'chilli sauce': ['tomato sauce (330 liters per liter)', 'garlic sauce (280 liters per liter)', 'yogurt sauce (250 liters per liter)'],
    'turnip soup': ['parsnip soup (250 liters per liter)', 'potato soup (200 liters per liter)', 'cauliflower soup (150 liters per liter)'],
    'corn tortilla': ['wheat tortilla (500 liters per kilogram)', 'rice tortilla (400 liters per kilogram)', 'almond flour tortilla (300 liters per kilogram)'],
    'sweetcorn fritters': ['zucchini fritters (400 liters per kilogram)', 'potato fritters (300 liters per kilogram)', 'carrot fritters (350 liters per kilogram)'],
    'sweet potato fries': ['carrot fries (250 liters per kilogram)', 'zucchini fries (200 liters per kilogram)', 'pumpkin fries (300 liters per kilogram)'],
    'paprika seasoning': ['chili powder (250 liters per kilogram)', 'cumin powder (200 liters per kilogram)', 'turmeric powder (180 liters per kilogram)'],
    'jalapeño poppers': ['stuffed bell peppers (400 liters per kilogram)', 'stuffed zucchini (350 liters per kilogram)', 'stuffed mushrooms (300 liters per kilogram)'],
    'ginger tea': ['mint tea (500 liters per liter)', 'chamomile tea (400 liters per liter)', 'lemon balm tea (350 liters per liter)'],
    'garlic bread': ['herb bread (1000 liters per kilogram)', 'olive bread (900 liters per kilogram)', 'tomato bread (800 liters per kilogram)'],
    'pea soup': ['lentil soup (1000 liters per liter)', 'chickpea soup (900 liters per liter)', 'split pea soup (800 liters per liter)'],
    'eggplant parmesan': ['zucchini parmesan (300 liters per kilogram)', 'mushroom parmesan (250 liters per kilogram)', 'cauliflower parmesan (200 liters per kilogram)'],
    'banana bread': ['pumpkin bread (1500 liters per kilogram)', 'zucchini bread (1200 liters per kilogram)', 'carrot bread (1000 liters per kilogram)'],
    'apple crumble': ['peach crumble (1500 liters per kilogram)', 'plum crumble (1200 liters per kilogram)', 'rhubarb crumble (1300 liters per kilogram)'],
    'pear tart': ['apple tart (1300 liters per kilogram)', 'apricot tart (1100 liters per kilogram)', 'cherry tart (1200 liters per kilogram)'],
    'grape jelly': ['apple jelly (500 liters per kilogram)', 'blueberry jelly (600 liters per kilogram)', 'raspberry jelly (650 liters per kilogram)'],
    'orange marmalade': ['lemon marmalade (1000 liters per kilogram)', 'grapefruit marmalade (1100 liters per kilogram)', 'lime marmalade (950 liters per kilogram)'],
    'kiwi jam': ['strawberry jam (400 liters per kilogram)', 'raspberry jam (500 liters per kilogram)', 'blackberry jam (550 liters per kilogram)'],
    'watermelon sorbet': ['lemon sorbet (250 liters per liter)', 'mango sorbet (350 liters per liter)', 'pineapple sorbet (300 liters per liter)'],
    'pomegranate salad dressing': ['balsamic dressing (500 liters per liter)', 'lemon dressing (300 liters per liter)', 'yogurt dressing (250 liters per liter)'],
    'pineapple upside-down cake': ['peach upside-down cake (1500 liters per kilogram)', 'apple upside-down cake (1300 liters per kilogram)', 'plum upside-down cake (1400 liters per kilogram)'],
    'mango chutney': ['peach chutney (1500 liters per kilogram)', 'plum chutney (1200 liters per kilogram)', 'apricot chutney (1300 liters per kilogram)'],
    'cucumber pickles': ['radish pickles (150 liters per kilogram)', 'beetroot pickles (170 liters per kilogram)', 'carrot pickles (150 liters per kilogram)'],
    'carrot cake': ['zucchini cake (1500 liters per kilogram)', 'pumpkin cake (1300 liters per kilogram)', 'banana cake (1000 liters per kilogram)'],
    'stuffed onion': ['stuffed bell pepper (400 liters per kilogram)', 'stuffed tomato (350 liters per kilogram)', 'stuffed mushroom (300 liters per kilogram)'],
    'potato chips': ['carrot chips (250 liters per kilogram)', 'zucchini chips (200 liters per kilogram)', 'beetroot chips (180 liters per kilogram)']

}

# Growing regions in India
growing_regions_data = {
    # Fruits
    'banana': ['Tamil Nadu', 'Maharashtra', 'Gujarat', 'Andhra Pradesh'],
    'apple': ['Himachal Pradesh', 'Jammu & Kashmir', 'Uttarakhand'],
    'pear': ['Punjab', 'Haryana', 'Uttarakhand'],
    'grapes': ['Maharashtra', 'Karnataka', 'Tamil Nadu'],
    'orange': ['Maharashtra', 'Madhya Pradesh', 'Assam', 'Nagaland'],
    'kiwi': ['Arunachal Pradesh', 'Himachal Pradesh', 'Meghalaya', 'Nagaland', 'Sikkim'],
    'watermelon': ['Karnataka', 'Maharashtra', 'Tamil Nadu', 'Rajasthan'],
    'pomegranate': ['Maharashtra', 'Gujarat', 'Karnataka', 'Andhra Pradesh'],
    'pineapple': ['Assam', 'West Bengal', 'Tripura', 'Kerala', 'Meghalaya'],
    'mango': ['Uttar Pradesh', 'Andhra Pradesh', 'Telangana', 'Maharashtra', 'Karnataka'],
    # Vegetables
    'cucumber': ['Assam', 'West Bengal', 'Karnataka', 'Madhya Pradesh'],
    'carrot': ['Himachal Pradesh', 'Punjab', 'Karnataka', 'Uttar Pradesh'],
    'capsicum': ['Himachal Pradesh', 'Madhya Pradesh', 'Karnataka', 'Punjab'],
    'onion': ['Maharashtra', 'Karnataka', 'Gujarat', 'Madhya Pradesh', 'Odisha'],
    'potato': ['Uttar Pradesh', 'West Bengal', 'Bihar', 'Punjab'],
    'lemon': ['Andhra Pradesh', 'Maharashtra', 'Gujarat', 'Tamil Nadu'],
    'tomato': ['Madhya Pradesh', 'Andhra Pradesh', 'Odisha', 'Karnataka', 'Maharashtra'],
    'radish': ['Haryana', 'Punjab', 'Uttar Pradesh', 'West Bengal'],
    'beetroot': ['Karnataka', 'Maharashtra', 'Uttar Pradesh', 'Himachal Pradesh'],
    'cabbage': ['West Bengal', 'Bihar', 'Odisha', 'Assam', 'Karnataka'],
    'lettuce': ['Himachal Pradesh', 'Uttarakhand', 'Jammu & Kashmir'],
    'spinach': ['West Bengal', 'Haryana', 'Uttar Pradesh', 'Punjab'],
    'soy bean': ['Madhya Pradesh', 'Maharashtra', 'Rajasthan'],
    'cauliflower': ['Bihar', 'Uttar Pradesh', 'West Bengal', 'Madhya Pradesh'],
    'bell pepper': ['Himachal Pradesh', 'Karnataka', 'West Bengal', 'Madhya Pradesh'],
    'chilli pepper': ['Andhra Pradesh', 'Madhya Pradesh', 'Karnataka', 'Odisha'],
    'turnip': ['Punjab', 'Himachal Pradesh', 'Haryana', 'Jammu & Kashmir'],
    'corn': ['Madhya Pradesh', 'Karnataka', 'Rajasthan', 'Uttar Pradesh'],
    'sweetcorn': ['Karnataka', 'Maharashtra', 'Andhra Pradesh', 'Tamil Nadu'],
    'sweet potato': ['Odisha', 'West Bengal', 'Uttar Pradesh', 'Bihar'],
    'paprika': ['Madhya Pradesh', 'Karnataka', 'West Bengal', 'Uttar Pradesh'],
    'jalepeño': ['Maharashtra', 'Karnataka', 'West Bengal', 'Odisha'],
    'ginger': ['Assam', 'Kerala', 'Meghalaya', 'Sikkim'],
    'garlic': ['Madhya Pradesh', 'Gujarat', 'Rajasthan', 'Uttar Pradesh'],
    'peas': ['Punjab', 'Haryana', 'Uttar Pradesh', 'West Bengal'],
    'eggplant': ['West Bengal', 'Odisha', 'Gujarat']
    
}

# Function to translate content
def translate_text(text, language):
    if language == "English":
        return text
    return translator.translate(text, dest=language_dict[language]).text

# Sidebar for language selection
st.sidebar.title("Language Selection")
language = st.sidebar.selectbox("Choose Language", list(language_dict.keys()))

# Sidebar for app mode selection
st.sidebar.title(translate_text("Dashboard", language))
app_mode = st.sidebar.selectbox(translate_text("Select Page", language), 
                                [translate_text("Home", language), 
                                 translate_text("About Project", language), 
                                 translate_text("Prediction", language), 
                                 translate_text("Final Product Water Footprint", language)])

# Main Page
if app_mode == translate_text("Home", language):
    st.header(translate_text(" calculation of Water Footprints for different Agricultural Products", language))
    image_path = "home_img.jpg"
    st.image(image_path)

elif app_mode == translate_text("About Project", language):
    st.header(translate_text("About Project", language))
    st.subheader(translate_text("About Dataset", language))
    st.text(translate_text("This dataset contains images of the following food items:", language))
    st.code(translate_text("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.", language))
    st.code(translate_text("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.", language))

elif app_mode == translate_text("Prediction", language):
    st.header(translate_text("Model Prediction", language))
    test_image = st.file_uploader(translate_text("Choose an Image:", language), type=["jpg", "png"])

    if test_image is not None:
        st.image(test_image, use_column_width=True)

        if st.button(translate_text("Predict", language)):
            with st.spinner(translate_text('Predicting...', language)):
                result_index = model_prediction(test_image)

            with open("labels.txt") as f:
                content = f.readlines()
            labels = [line.strip() for line in content]

            predicted_label = labels[result_index]
            st.success(translate_text(f"Model is predicting it's a {predicted_label}", language))

            water_footprint = water_footprint_data.get(predicted_label.lower(), translate_text("No data available", language))
            st.info(translate_text(f"Water Footprint for {predicted_label}: {water_footprint}", language))

            health_benefit = health_benefits_data.get(predicted_label.lower(), translate_text("No health benefits data available", language))
            st.info(translate_text(f"Health Benefits of {predicted_label}: {health_benefit}", language))

            if predicted_label.lower() in alternative_items:
                alternatives = ", ".join(alternative_items[predicted_label.lower()])
                st.info(translate_text(f"Consider these alternative products with lower water footprints: {alternatives}", language))

            growing_regions = growing_regions_data.get(predicted_label.lower(), translate_text("No data available", language))
            if growing_regions != translate_text("No data available", language):
                regions_list = ", ".join(growing_regions)
                st.info(translate_text(f"Regions in India where {predicted_label} is grown: {regions_list}", language))

elif app_mode == translate_text("Final Product Water Footprint", language):
    st.header(translate_text("Final Product Water Footprint", language))
    search_product = st.text_input(translate_text("Enter a Final Product Name", language), "")

    if search_product:
        water_footprint = final_product_footprint_data.get(search_product.lower(), translate_text("No data available", language))
        st.info(translate_text(f"Water Footprint for {search_product}: {water_footprint}", language))

        health_benefit = health_benefits_final_product.get(search_product.lower(), translate_text("No health benefits data available", language))
        st.info(translate_text(f"Health Benefits of {search_product}: {health_benefit}", language))

        if search_product.lower() in alternative_final_products:
            alternatives = ", ".join(alternative_final_products[search_product.lower()])
            st.info(translate_text(f"Consider these alternative final products with lower water footprints: {alternatives}", language))
