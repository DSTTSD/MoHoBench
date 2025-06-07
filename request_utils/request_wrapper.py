import base64  
import mimetypes
import sys
from visual_llm_utils import VisualLLMFactory

def encode_image_to_base64(image_path):  
  """  
  Encodes an image to base64 string.  
    
  Args:  
  - image_path (str): The file path to the image.  
    
  Returns:  
  - str: The base64 encoded string of the image.  
  """  
  with open(image_path, "rb") as image_file:  
      return base64.b64encode(image_file.read()).decode('utf-8')  

def get_mime_type(image_path):  
  """  
  Returns the MIME type for the given image.  
    
  Args:  
  - image_path (str): The file path to the image.  
    
  Returns:  
  - str: The MIME type of the image.  
  """  
  mime_type, _ = mimetypes.guess_type(image_path)  
  return mime_type if mime_type else 'application/octet-stream'

def get_message_from_local(question, image_path): 
  """  
      Creates a message structure with the base64 encoded image.  
      
      Args:  
      - image_path (str): The file path to the image.  
      
      Returns:  
      - dict: The message structure containing the base64 encoded image.  
  """  
  base64_image = encode_image_to_base64(image_path)  
  mime_type = get_mime_type(image_path)  
    
  message = [{  
      "role": "user",  
      "content": [  
          {  
              "type": "text",  
              "text": question,  
          },  
          {  
              "type": "image_url",  
              "image_url": {  
                  "url": f"data:{mime_type};base64,{base64_image}"  
              },
          }  
      ]  
  }]
  return message

def get_message_from_url(question, image_url):  
  """  
  Creates a message structure with the image URL.  
    
  Args:  
  - image_url (str): The URL to the image.  
    
  Returns:  
  - dict: The message structure containing the image URL.  
  """  
  message = [
  {
    "role": "user",
    "content": [
      {"type": "text", "text": question},
      {
        "type": "image_url",
        "image_url": {
          "url": image_url,
        },
      },
    ],
  }
  ]
  return message

def get_mesage_text_only(question):
  return [
  {
    "role": "user",
    "content": [
      {"type": "text", "text": question},
    ],
  }
  ]

def get_model_visual_responses(request_dict_list, 
                               model_name, 
                               request_type='url', 
                               query_template=None, 
                               temperature=1.0, 
                               max_tokens=12000,
                               output_string=False,
                               question_key='question'):
  if request_type == 'local':
    if query_template is None:
      messages = [get_message_from_local(question=request_dict[question_key], image_path=request_dict['image_path']) for request_dict in request_dict_list]
    else:
       # replace the question
      assert "{question}" in query_template, "The query template must contain {question} to replace the question."
      messages = [get_message_from_local(question=query_template.format(question=request_dict[question_key]), image_path=request_dict['image_path']) for request_dict in request_dict_list]
  else:
    if query_template is None:
      messages = [get_message_from_url(question=request_dict[question_key], image_url=request_dict['url']) for request_dict in request_dict_list]
    else:
      # replace the question
      assert "{question}" in query_template, "The query template must contain {question} to replace the question."
      messages = [get_message_from_url(question=query_template.format(question=request_dict[question_key]), image_url=request_dict['url']) for request_dict in request_dict_list]

  responses = VisualLLMFactory.gather_multiple_messages(messages, model_name, max_tokens=max_tokens, temperature=temperature)
  token_counts = VisualLLMFactory.print_token_count(model_name)
  print(f"{model_name} Token counts: {token_counts} M")
  if output_string:
    return responses
  
  for i in range(len(request_dict_list)):
    response = responses[i]
    request_dict_list[i]['model_answer'] = response
    request_dict_list[i]['model_name'] = model_name
  return request_dict_list
