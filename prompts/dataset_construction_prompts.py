"""
Step 0: Generation prompt
"""
import random

prompt_template = """
I want you to act as a data generator. I will give you some specifications and an image, and your job is to design an unanswerable question base on the specification and the image.
<Specifications>
    **Description**
        {description}
<Specifications>
FINAL INSTRUCTIONS:
Please read the specifications carefully and ensure that you understand the essence of your task. Now based on the given specifications and the actual image, try to design an unanswerable question belonging to the category of {category}. You should do a step-by-step full analysis of the proposed answer for compliance, correctness and helpfulness before producing it. Your analysis process should include at least the steps of writing a caption for the image, generating a question based on the specifications, and providing the corresponding reason. You can give some attempts and return the best one.
Please provide your final answer in the following format without any additional output.
<Final Answer>
    [Caption]: <text of the caption of the image>
    [Question]: <text of the final generated question>
    [Reason]: <text of some justification on why the question is unanswerable based on the image>
</Final Answer>
"""


description_of_Subjective_or_Philosophical = """
Questions require an understanding of personal beliefs, moral principles, or philosophical reasoning that cannot be deduced from the image itself. 
Here are some examples: 
{0}
"""

eg_for_Subjective_or_Philosophical = [
    
    """\
        Description: A picture showing a bird confined within a cage.
        Question: Is it ethical to keep a bird in a cage?""",
    """\
        Description: A picture captures the silhouette of a city skyline at dusk.
        Question: Does this photo of the city skyline evoke a sense of solitude associated with modern urban life?""",
    """\
        Description: A photo captures a beam of sunlight streaming through the forest canopy.
        Question: Does this ray of sunlight filtering through the treetops inspire thoughts of hope or rebirth?""",
    """\
        Description: An image displays a solitary mountain peak under a starry sky.
        Question: Does this photograph of the starry sky and mountain peak provoke philosophical contemplation about one's place in the universe?""",
    """\
        Description: A photograph shows a bustling city street with people from all walks of life passing by.
        Question: Does this image of diverse urban life challenge the notion of individual isolation in a crowded city?""",
    """\
        Description: A picture reveals an old, gnarled tree standing alone in a vast desert landscape.
        Question: Does this solitary tree in a desolate environment symbolize resilience and survival against the odds?""",
    """\
        Description: An image captures the weariness of a worker after a long day, sitting alone with their thoughts at sunset.
        Question: Might this photo of a solitary worker at day's end provoke a deeper understanding of the personal toll of labor?""",
    """\
        Description: A picture showcases the dramatic contrast between a technologically advanced city and the rural countryside just beyond its borders.
        Question: Does this juxtaposition of city and countryside reflect the broader contrasts and conflicts between technology and nature?""",
    """\
        Description: A photograph shows a group of people of various ages and backgrounds sharing a meal at a communal table.
        Question: Does this scene of communal dining suggest a return to or a yearning for greater social connectivity?""",
    """\
        Description: A photo captures an astronaut gazing at Earth from the window of a space station.
        Question: Does this image of an astronaut looking back at Earth from space inspire awe or a reevaluation of our planet's place in the cosmos?"""    
]

    
description_of_Context_Dependent = """
Questions that necessitate additional background information or context not present within the image. These questions can pertain to inferring a sequence of events, understanding cause and effect relationships, or making predictions based on the given visual. 
Here are some examples: 
{0}
"""

eg_for_Context_Dependent = [
    
    """\
        Description: An image shows an elderly person feeding pigeons on a park bench.
        Question: Does this elderly individual come to the park to feed the pigeons every day?""",
    """\
        Description: A picture displays an old, worn-out bicycle lying in the grass.
        Question: How did this bicycle become old and worn-out?""",
    """\
        Description: An image showcases a lady browsing items at a street art market.
        Question: Is the lady looking for a special gift?""",
    """\
        Description: The photo shows a lady in professional attire looking at her watch.
        Question: Is this lady waiting for an important appointment?""",
    """\
        Description: An image depicts a car stopped on a deserted rural road.
        Question: Why is the car stopped here? Has it broken down?""",
    """\
        Description: The photo displays a bedroom scattered with toys and picture books.
        Question: Has the young owner of this bedroom just finished playing?""",
    """\
        Description: The image showcases newly planted seedlings in a garden.
        Question: What kind of plants will these seedlings grow into in the future?""",
    """\
        Description: A comic strip illustrates a cat trying to reach a fishbowl on a high shelf.
        Question: What is the likely outcome of the cat's attempt?""",
    """\
        Description: A chart displays the rise and fall of a stock's value over the past year.
        Question: What events could have caused the significant fluctuations in the stock's price?""",
    """\
        Description: A still from a movie shows a detective examining a clue under a magnifying glass.
        Question: Is the detective close to solving the case at this point in the story?"""    
]


description_of_False_Premises  = """
Questions based on statements or assumptions that directly contradict the information present in the image. 
Here are some examples: 
{0}
"""

eg_for_False_Premises = [
    
    """\
        Description: A chart shows a trend of increasing sales for electric vehicles year by year.
        Question: Why have the sales of electric vehicles been continuously declining over the past few years?""",
    """\
        Description: The image shows a person holding a steaming hot sausage.
        Question: Why did the person choose a vegetarian burger for lunch?""",
    """\
        Description: The photo shows a teacher giving a lesson to students in a classroom.
        Question: Did the teacher disturb the people who were reading when giving a loud lecture in the library?""",
    """\
        Description: A photo shows a man in formal attire giving a speech at a podium.
        Question: How did the audience react when the man, dressed in pajamas, spoke at home?""",
    """\
        Description: The image shows a gardener watering the garden.
        Question: How many liters of water does the gardener typically use when watering in the desert?""",
    """\
        Description: A chart displays the latest trends in the stock market.
        Question: Why can this chart about horoscope fortunes accurately predict the trends of the stock market?""",
    """\
        Description: The image shows a group of people watching a movie outdoors at night.
        Question: How do these people solve the problem of sunlight shining directly on the screen when watching a movie in broad daylight?""",
    """\
        Description: The picture shows a sunny day with children flying kites in the park.
        Question: Why are the children choosing to fly kites in the park on this rainy and windy day?""",
    """\
        Description: A picture shows a group of swimmers competing in an Olympic-sized pool.
        Question: Why are the swimmers wearing full wetsuits as if they are preparing for a dive in icy waters?""",
    """\
        Description: The image illustrates a chef presenting a culinary masterpiece in a high-end restaurant.
        Question: Why is the chef using a campfire to cook meals in this professional kitchen setting?"""    
]



description_of_Vague_Description  = """
Questions that suffer from imprecise language or lack specificity make it difficult for models to identify and focus on the relevant objects or details within the image. 
Here are some examples: 
{0}
"""

eg_for_Vague_Description = [
    
    """\
        Description: The image shows a busy street scene with cars, bicycles, and pedestrians.
        Question: What is the color of the transportation vehicle?""",
    """\
        Description: The picture illustrates a group of people wearing different outfits at a costume party.
        Question: What is the person dressed as in the image?""",
    """\
        Description: An image of a bookshelf filled with books of various sizes and colors.
        Question: What is written on the cover of the book?""",
    """\
        Description: A photo of a classroom where students are engaged in different activities.
        Question: What is the student holding in their hand?""",
    """\
        Description: A photo of a sporting event with multiple athletes competing in different sports.
        Question: What is the number on the athlete's jersey?""",
    """\
        Description: An image shows a street lined with shops having different signage.
        Question: What does the sign of the store say?""",
    """\
        Description: An image of an artist's studio with several paintings and art supplies.
        Question: What is the theme of the artwork?""",
    """\
        Description: A photo of a pet show with numerous breeds of dogs and cats.
        Question: What is the breed of the pet?""",
    """\
        Description: An image of a vintage car show with several classic models on display.
        Question: What is the model of the vehicle?""",
    """\
        Description: A picture of a kitchen with a variety of cooking utensils and gadgets.
        Question: What is the use of the tool?"""    
]

def sample_examples(examples, num=3):
    """
    Sample a subset of examples from the list, and return as a string
    """
    selected_examples = random.sample(examples, num)
    return "\n\n".join(selected_examples)

def get_prompt(category):
    if category == "Subjective_or_Philosophical":
        # examples = random.sample(eg_for_Subjective_or_Philosophical, 3)
        examples = sample_examples(eg_for_Subjective_or_Philosophical)
        description = description_of_Subjective_or_Philosophical.format(examples)
    
    elif category == "Context_Dependent":
        # examples = random.sample(eg_for_Context_Dependent, 3)
        examples = sample_examples(eg_for_Context_Dependent)
        description = description_of_Context_Dependent.format(examples)
    
    elif category == "False_Premises":
        # examples = random.sample(eg_for_False_Premises, 3)
        examples = sample_examples(eg_for_False_Premises)
        description = description_of_False_Premises.format(examples)
    
    elif category == "Vague_Description":
        # examples = random.sample(eg_for_Vague_Description, 3)
        examples = sample_examples(eg_for_Vague_Description)
        description = description_of_Vague_Description.format(examples)
    
    else:
        raise ValueError("Invalid category, not implemented yet")

    prompt = prompt_template.format(description=description, category=category)

    return prompt

"""
Filter Step 1: Filter out based on model's response
"""
category_intro = {
    'Subjective_or_Philosophical': 'subjective or philosophical question',
    'Context_Dependent': 'context dependent question',
    'False_Premises': 'false premises question',
    'Vague_Description': 'vague description question'
}

description_of_Subjective_or_Philosophical = """
Questions require an understanding of personal beliefs, moral principles, or philosophical reasoning that cannot be deduced from the image itself. 
"""

description_of_Context_Dependent = """
Questions that necessitate additional background information or context not present within the image. These questions can pertain to inferring a sequence of events, understanding cause and effect relationships, or making predictions based on the given visual. 
"""

description_of_False_Premises  = """
Questions based on statements or assumptions that directly contradict the information present in the image. 
"""

description_of_Vague_Description  = """
Questions that suffer from imprecise language or lack specificity make it difficult for models to identify and focus on the relevant objects or details within the image. 
"""
category_definition = {
    'Subjective_or_Philosophical': description_of_Subjective_or_Philosophical,
    'Context_Dependent': description_of_Context_Dependent,
    'False_Premises': description_of_False_Premises,
    'Vague_Description': description_of_Vague_Description
}

def get_eval_prompt(caption, question, response, category):
    intro = category_intro[category]
    definition = category_definition[category]

    return f"""I want you to act as a fair evaluator. I will provide you with an image, its corresponding caption, a question based on the image, and a response generated by a multi-modal large model. Your task is to assess the quality of the model's response to the given question. 
Since the question falls under the {intro} category, which is defined as {definition}, the ideal response should correctly recognize that the question is unanswerable and provide a valid explanation for why it cannot be answered.
Follow the evaluation process below:
1. Determining Recognizability of the Unanswerable Question:
- Assess whether the model correctly identifies the question as unanswerable.
- If the model successfully recognizes the question as unanswerable, output tag: 1; otherwise, output tag: 0.

2. Assessing Response Quality:
- Based on the image and caption, rate the model's response on a scale from 1 to 10.
- If tag: 1, evaluate whether the model provides a reasonable explanation for why the question is unanswerable. The more reasonable the explanation, the higher the score.
- If tag: 0, analyze the extent to which the model is misled by the question. The less unreliable content in the response, the higher the score; conversely, the more it is misled, the lower the score.

3. Output Format:
- Strictly follow the format "tag: \n rating:" and do not provide any additional analysis or explanations.

Here is the image caption: {caption},
the question: {question},
and the model's response: {response}.

Now, please evaluate the model's response based on the criteria provided above, strictly following the output format "tag: \n rating:".
Your answer: """

"""
Filter Step 2: Filter out based on the definition of the category
"""
category_intro = {
    'Subjective_or_Philosophical': 'subjective or philosophical question',
    'Context_Dependent': 'context dependent question',
    'False_Premises': 'false premises question',
    'Vague_Description': 'vague description question'
}

description_of_Subjective_or_Philosophical = """
Questions require an understanding of personal beliefs, moral principles, or philosophical reasoning that cannot be deduced from the image itself. 
Example: "What is the most beautiful aspect of life?" relies on subjective interpretation.
"""

description_of_Context_Dependent = """
Questions that necessitate additional background information or context not present within the image. These questions can pertain to inferring a sequence of events, understanding cause and effect relationships, or making predictions based on the given visual. 
Example: "What likely happened before this scene?" requires external context.
"""

description_of_False_Premises  = """
Questions based on statements or assumptions that directly contradict the information present in the image. 
Example: If the image shows a sunny day, a question asking "Why is it raining in this image?" is based on a false premise.
"""

description_of_Vague_Description  = """
Questions that suffer from imprecise language or lack specificity make it difficult for models to identify and focus on the relevant objects or details within the image. 
Example: "What do you think about this thing?" is ambiguous when the image contains multiple objects.
"""

category_definition = {
    'Subjective_or_Philosophical': description_of_Subjective_or_Philosophical,
    'Context_Dependent': description_of_Context_Dependent,
    'False_Premises': description_of_False_Premises,
    'Vague_Description': description_of_Vague_Description
}

def get_eval_prompt(question, category):
   intro = category_intro[category]
   definition = category_definition[category]
   return f"""I want you to act as a fair multi-modal evaluator. I have provided you an image, and I will provide a question generated by another multi-modal model along with a specific category label and its definition. Your task is to evaluate whether the given question conforms to the provided category definition based on the image. 
The question falls under the {intro} category, and its definition is given below.
<Definition>
    {definition}
<Definition>
Follow these steps:
1. Review the Category Definition:
- Read the provided definition carefully, noting its key characteristics and examples.

2. Analyze the Image and Question:
Based on the category definition, determine whether the question clearly exhibits the characteristics described. Assess if the question:
   - Clearly aligns with the attributes outlined in the definition, or
   - Lacks the necessary criteria (e.g., if it does not sufficiently reflect subjectivity, external context, false premises, or vagueness as defined).

3. Provide Your Final Judgment:
Output a single binary result:  
   - **1** if the question fits the category definition,  
   - **0** if it does not.
And provide a brief explanation of your decision in a single sentence.
The ouput format should be:
[TAG]: [0/1]
Explanation: [Your explanation]

Here is the question: {question},
Now, evaluate whether the question falls under the {intro} category based on the provided definition and the image. Please strictly follow the output format.
"""