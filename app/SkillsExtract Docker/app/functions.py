import pandas as pd
import re
import spacy
from spacy.tokenizer import Tokenizer
import aspose.words as aw

# load pre-trained model
nlp = spacy.load('en_core_web_sm')
# Tokenize words only with the whitespace rule
# N-grams will no longer be treated as 'N' and '-grams'
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)

def read_resume(resume_filename):
    resume = aw.Document(resume_filename)
    resume_string = resume.to_string(aw.SaveFormat.TEXT).split('\r\n')
    resume_string = ' '.join(resume_string[1:-3])
    return resume_string

def preprocess(txt):
    txt = txt.lower()
    # these must come first
    txt = re.sub('b\.\S*', '', txt) # remove all bachelor qualifications
    txt = re.sub('m\.\S*', '', txt) # remove all master qualifications
    # then these
    txt = txt.replace("'","").replace("’","") # remove apostrophes
    txt = re.sub('<.*?>',' ',txt) # remove <> tags
    txt = re.sub('http\S+\s*', ' ', txt)  # remove URLs
    txt = re.sub('RT|cc', ' ', txt)  # remove RT and cc
    txt = re.sub('#\S+', '', txt)  # remove hashtags
    txt = re.sub('@\S+', '  ', txt)  # remove mentions
    txt = re.sub('[^a-zA-Z]', ' ', txt) # Remove non-English characters
    txt = re.sub('\s+', ' ', txt)  # remove extra whitespace

    # tokenize word
    txt = nlp(txt)

    # remove stop words
    txt = [token.text for token in txt if not token.is_stop]

    return ' '.join(txt)

def n_grams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def generate_list_of_skills(text):
    nlp_text = nlp(text)

    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]
    # all the resume skills will be saved here
    skillset = []
        
    # check for one-grams (example: python)
    for token in tokens:
        skillset.append(token)
        
    # check for noun_chunks (example: machine learning)
    for token in nlp_text.noun_chunks:
        token = token.text.lower().strip()
        skillset.append(token)

    # check for N-grams that SpaCy missed in the noun_chuncks
    for n in range(2, 10):
        for token in n_grams(tokens, n):
            token = token.lower().strip()
            skillset.append(token)

    return skillset

def extract_skills(resume_text, skills_api, clean=True):
    if clean == True:
        resume_text = preprocess(resume_text)
    
    skillset = set([i for i in set([i.lower() for i in generate_list_of_skills(resume_text)])])
    return skillset.intersection(skills_api)

def extract_skills(resume_text, skills_api, clean=True):
    if clean == True:
        resume_text = preprocess(resume_text)
    
    skillset = set([i for i in set([i.lower() for i in generate_list_of_skills(resume_text)])])
    return skillset.intersection(skills_api)

def section_break(original_resume_text):
    pattern = r'\b[A-Z]+\b'
    # find all the words that are fully uppercased
    res = list(re.finditer(pattern, original_resume_text))

    # find only those uppercased words that are also NOUN
    res = [x for x in res if nlp(x.group())[0].pos_ == 'NOUN']
    
    ans = []

    # if there is no uppercased NOUN
    if len(res) == 0:
        ans.append(original_resume_text)
    # if there is just one uppercased NOUN
    elif len(res) == 1:
        ans.append(original_resume_text[res[0].span()[1]:])
    else:
        i = 1
        while i < len(res):
            ans.append(original_resume_text[res[i-1].span()[1]:res[i].span()[0]])
            i += 1
        ans.append(original_resume_text[res[i-1].span()[1]:])
    return ans

def date_search(resume):
    ans = []

    # find all the date occurrence based on the regular expression
    pattern = r'(((Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Nov(ember)?|Dec(ember)?)|(\d{1,2}\s?\/){0,2}|(\d{1,2}\s?\-){0,2})\s?[-/ ]?\s?\d{4}?)|\bPresent\b|\bpresent\b|\bCurrent\b|\bcurrent\b'
    res = list(re.finditer(pattern, resume))

    if len(res) > 1:
        for ele in res:
            # this is to eradicate the results of having only year but without month
            if len(ele.group().strip()) > 5:
                ans.append([ele.start(), ele.end(), ele.group().strip()])

    res = []
    # Convert "present" and "current" to today's date
    for ele in ans:
        if ele[2].lower() == 'present' or ele[2].lower() == 'current':
            today = pd.to_datetime('today').date()
            ele[2] = today
            res.append(ele)
        else:
            # catch DateParse Error here
            try:
                day = pd.to_datetime(ele[2]).date()
                ele[2] = day
                res.append(ele)
            except :
                print('Cannot parse the date: ', ele[2])

    # all the date results are given in the form of [datetime_start_index, datetime_end_index, datetime]
    return res

def experience_tagging(date_list):
    i = 1
    cleaned_section = {}

    while i < len(date_list):

        prev = date_list[i-1]
        cur = date_list[i]
        
        if cur[0] < prev[1] + 10:
            # Taking ceiling of the year of experience
            key = ((cur[2] - prev[2]).days // 365) + 1
            
            # section starts at the (end index of the current date) + 1
            frm = cur[1]+1

            if i < len(date_list) - 1:
                # if there is another date that appears later, then the section will be until the start index of the next date
                until = date_list[i+1][0]
            else:
                # else the section will be until the end of the resume
                until = -1
            
            # Multiple projects with same year of experience, we do 'chaining' here
            if key in cleaned_section:
                cleaned_section[key].append((frm, until))
            else:
                cleaned_section[key] = [(frm, until)]
            i += 2

        else:
            # ignore the current date, possibly it is useless
            i += 1

    return cleaned_section

def skills_experience_level_identification(resume, skills_api, clean=True):
    res = {}
    sections = section_break(resume)

    for section in sections:
        date_list = date_search(section)
        experience_sec = experience_tagging(date_list)
        for key in experience_sec:
            for start, end in experience_sec[key]:
                skills_list = extract_skills(section[start:end], skills_api, clean)
                for ele in skills_list:
                    if ele not in res:
                        res[ele] = key
                    else:
                        res[ele] = max(key, res[ele])

    # for all the skills which do not have any level of experience, assign a default value of 1
    skills_list = extract_skills(resume, skills_api, True)
    for ele in skills_list:
        if ele not in res:
            res[ele] = 1

    return dict(sorted(res.items(), key = lambda x:x[1], reverse = True))