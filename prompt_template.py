class PromptTemplate:
    intent_datasets = ['banking', 'clinc', 'bankingKIR', 'clincKIR', 'mtop', 'massive']
    emo_datasets = ['emo']
    stackoverflow_datasets = ['stackoverflow', 'stackoverflowKIR']
    topic_datasets  = ['arxiv', 'reddit']
    @staticmethod
    def render_initial_prompt(dataset_name,target):
        if dataset_name in PromptTemplate.intent_datasets:
            template = (
                        "Task Instructions:\n"
                        
                        "1. You are given multiple target utterances, which are unlabeled but all express the same intent.\n"                 
                        "2. Create one intent that best fits all target utterances.\n\n"
                
                        "Answer Format:\n"
                        "1. Use the format: The intent is: [intent]\n"
                        "2. Use lowercase and separate all words with underscores (e.g., bookaflight → book_a_flight).\n"
                        "3. No explanations. Stick to the format.\n\n"
                        
                        "Task:\n"                       
                        "Target Utterances:\n"
                        "{{target_str}}"         
                        ) 

            target_str = "\n".join([f"{i+1}. {example.strip().capitalize()}"  for i, example in enumerate(target)])

            return template.replace("{{target_str}}", target_str)

    
    @staticmethod
    def render_sent_selection_prompt(dataset_name,target,demo_examples):
        demo_str = "\n".join([f"{i+1}. {example.strip().capitalize()}" for i, example in enumerate(demo_examples)])   
        target_str = target.capitalize()
        if dataset_name in PromptTemplate.intent_datasets:
            template = (
                "Task Instructions:\n\n"
                
               
                "Compare each Candidate Utterance to the Target Utterance. Select only Candidates with the same intent as the Target Utterance. Intent refers to the request or the purpose the user wants to achieve.\n\n"
                
                "Answer Format:\n"
                "1. Only provide the final selection of Candidate Utterances by listing their numbers.\n"
                "2. If Candidates 3, 4, and 9 match, write: The Candidate utterance number(s): 3, 4, 9\n"
                "3. If no Candidates match, write: The Candidate utterance number(s): none\n"
                "Note: Stick to the answer format and avoid providing extra explanations.\n\n"
                
                "Task:\n"
                "Target Utterance: {{target_str}}\n"
                "Candidate Utterances:\n{{demo_str}}\n"
            )
        elif dataset_name in PromptTemplate.emo_datasets:
            template = (
                "Task Instructions:\n\n"
                
                "Compare each Candidate Utterance to the Target Utterance. Select only Candidates with the same emotion as the Target Utterance.\n"
                
                "Answer Format:\n"
                "1. Only provide the final selection of Candidate Utterances by listing their numbers.\n"
                "2. If Candidates 3, 4, and 9 match, write: The Candidate utterance number(s): 3, 4, 9\n"
                "3. If no Candidates match, write: The Candidate utterance number(s): none\n"
                "Note: Stick to the answer format and avoid providing extra explanations.\n\n"
                
                "Task:\n"
                "Target Utterance: {{target_str}}\n"
                "Candidate Utterances:\n{{demo_str}}\n"
            )
        elif dataset_name in PromptTemplate.stackoverflow_datasets:
            template = (
                "Task Instructions:\n\n"
                
                "Compare each Candidate Question to the Target Question. Select only Candidates with the same main programming framework, language, tool, or concept as the Target Question.\n"
                
                "Answer Format:\n"
                "1. Only provide the final selection of Candidate Questions by listing their numbers.\n"
                "2. If Candidates 3, 4, and 9 match, write: The Candidate question number(s): 3, 4, 9\n"
                "3. If no Candidates match, write: The Candidate question number(s): none\n"
                "Note: Stick to the answer format and avoid providing extra explanations.\n\n"
                
                "Task:\n"
                "Target Question: {{target_str}}\n"
                "Candidate Questions:\n{{demo_str}}\n"
            )
        elif dataset_name in PromptTemplate.topic_datasets:
            template = (
                "Task Instructions:\n\n"
                
                "Compare each Candidate Text to the Target Text. Select only Candidates with the same topic as the Target Utterance.\n"
                
                "Answer Format:\n"
                "1. Only provide the final selection of Candidate Texts by listing their numbers.\n"
                "2. If Candidates 3, 4, and 9 match, write: The Candidate Text number(s): 3, 4, 9\n"
                "3. If no Candidates match, write: The Candidate Text number(s): none\n"
                "Note: Stick to the answer format and avoid providing extra explanations.\n\n"
                
                "Task:\n"
                "Target Text: {{target_str}}\n"
                "Candidate Texts:\n{{demo_str}}\n"
            )         
        return template.replace("{{target_str}}", target_str).replace("{{demo_str}}",demo_str)
    