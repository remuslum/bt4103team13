import gensim
import multiprocessing
import nltk
import numpy as np
import pandas as pd
import re
import sklearn
import spacy
import pickle
import boto3

from gensim import corpora
import aspose.words as aw
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import MultiLabelBinarizer
from spacy.tokenizer import Tokenizer

# load pre-trained model
nlp = spacy.load('en_core_web_sm')
# Tokenize words only with the whitespace rule
# N-grams will no longer be treated as 'N' and '-grams'
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)

def create_skills_api(skills_api_filename: str) -> set:
    skills_api = pd.read_excel(skills_api_filename)
    skills_api['name'] = skills_api['name'].apply(lambda x: re.sub("\W?\(.*?\)","",x))
    skills_api['name'] = skills_api['name'].apply(lambda x: x.strip().lower())
    skills_api['lemmatized_name'] = skills_api['name'].apply(lambda x: ' '.join([token.lemma_ for token in nlp(x)]))
    skills_api = set(skills_api['name']).union(set(skills_api['lemmatized_name']))
    return skills_api

def read_resume(resume_filename: str) -> str:
    resume = aw.Document(resume_filename)
    resume_string = resume.to_string(aw.SaveFormat.TEXT).split('\r\n')
    resume_string = ' '.join(resume_string[1:-3])
    return resume_string

def preprocess(txt: str) -> str:
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

def n_grams(tokens: list[str], n: int) -> list[str]:
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def generate_list_of_skills(text: str) -> list[str]:
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
    for n in range(2, 5):
        for token in n_grams(tokens, n):
            token = token.lower().strip()
            skillset.append(token)

    return skillset

def extract_skills(resume_text: str, skills_api: pickle, clean: bool=True) -> set:
    if clean == True:
        resume_text = preprocess(resume_text)
    
    # create a set of skills in lowercase from the resume
    skillset = set([i for i in set([i.lower() for i in generate_list_of_skills(resume_text)])])
    # find all valid skills using Skills API data
    return skillset.intersection(skills_api)

def section_break(original_resume_text: str) -> list[str]:
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

def date_search(resume: str) -> list[list]:
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

def experience_tagging(date_list: list[list]) -> list[tuple]:
    i = 1
    cleaned_section = {}

    while i < len(date_list):

        prev = date_list[i-1] # previous date
        cur = date_list[i] # current date
        
        # if current date is within 10 characters from the previous date
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

def skills_experience_level_identification(resume: str, skills_api: set, clean: bool=True) -> dict:
    res = {}
    # break down resume string into sections
    sections = section_break(resume)

    for section in sections:
        date_list = date_search(section)
        experience_sec = experience_tagging(date_list)
        for key in experience_sec:
            # for each section (`start` to `end`) that has the year of experience of `key`
            for start, end in experience_sec[key]:
                # find all the skills within the section
                skills_list = extract_skills(section[start:end], skills_api, clean)
                # for each skill, tag the maximum year of experience
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

    # return dictionary with sorted keys, `key` is the skill. `value` is the year of experience
    return dict(sorted(res.items(), key = lambda x:x[1], reverse = True))

def get_significance_table(df: pd.DataFrame, skills_dic: dict) -> pd.DataFrame:
    
    # creates a binary matrix indicating the presence of each skill
    df['Skills'] = df['ID'].apply(lambda x: list(skills_dic[x]))
    mlb = MultiLabelBinarizer()
    table = pd.DataFrame(mlb.fit_transform(df['Skills']),
                         columns=mlb.classes_,
                         index=df['Skills'].index)
    
    # add category column as y
    y = df['Category']
    table['y'] = y

    # sum by category column and divide by total number of instances
    agg_table = table.groupby(['y']).sum()
    agg_table = agg_table.T / table.groupby(['y']).size()

    # return a panda dataframe that has skills as rows, industry/job role as column
    return agg_table

def find_significant_skills(agg_table: pd.DataFrame) -> dict:

    # gauge skill levels according to percentiles
    skills_required = {}
    for col in agg_table.columns:
        # we only apply percentile method on skills that appear at least once
        skills = agg_table[col][agg_table[col] > 0]

        # no skills extracted > skills required = empty dictionary
        if len(skills) == 0: 
            skills_required[col] = dict()
        
        # some skills extracted > skills required = some dictionary
        else:
            series = agg_table[col][agg_table[col] >= np.percentile(skills, 95)]

            # if all skills above 95th percentile have same frequency, then scale them to 1 (max)
            if len(series.unique()) == 1:
                scaled_series = series.apply(lambda x: 1)
            # if skills above 95th percentile have different frequency, then scale to values between 0 and 1
            else:
                scaled_series = series.apply(lambda x: (x - series.min()) / (series.max() - series.min()))

            # bin skills according to percentiles
            binned_series = scaled_series.apply(lambda x: 5 if x > 0.7 else 4 if x > 0.3 else 3)
            
            # convert series to dictionary form
            skills_required[col] = binned_series.to_dict()

    # return a dictionary which `key` is the industry/job role and `value` is the dictionary with the corresponding competency level for each skill
    return skills_required

def create_skills_required_dictionary(df: pd.DataFrame, skills_api: set,clean: bool=True) -> dict:
    df["Skills"] = df["Text"].apply(lambda x: extract_skills(x,skills_api,clean))
    skills_dic = {}
    for index, row in df.iterrows():
        skills_dic[row.ID] = row.Skills
    table = get_significance_table(df,skills_dic)
    skills_required_dic = find_significant_skills(table)
    return skills_required_dic

# Map years of experience to competency level
def user_level_deduction(years: int) -> int:
    if years <= 2:
        return 3
    elif years <= 5:
        return 4
    else:
        return 5

# return skills gap grouped by skills
def skills_gap_identification(skills: dict, skills_required: dict) -> dict:
    diff = {}
    # compare the skill required vs the skill from resumes
    for key in skills_required:
        # if the applicant does not have the skill, then he needs to start picking up from level 3
        if key not in skills:
            diff[key] = [x for x in range(3, skills_required[key] + 1)]
        # if the applicant has the skill, find what is his competency level, then suggest him all the subsequent competency levels 
        else:
            user_level = user_level_deduction(skills[key])
            if user_level < skills_required[key]:
                diff[key] = [x for x in range(user_level + 1, skills_required[key] + 1)]
    # return a dictionary with `key` as the skill that need to be bridged and `value` as the difference in competency levels
    return diff

# group skills gap by level
def skills_gap_by_level(skills_gap: dict) -> dict:
    new_skills_gap = {}
    for skill in skills_gap:
        for level in skills_gap[skill]:
            if level in new_skills_gap:
                new_skills_gap[level].append(skill)
            else:
                new_skills_gap[level] = [skill]
    return new_skills_gap

def create_courses_dataset(courses_dataset_filename: str) -> pd.DataFrame:
    courses = pd.read_excel(courses_dataset_filename)
    courses = courses.fillna("")
    courses['Description'] = courses['jobFamily'] + " " \
                            + courses['Marketing Name'] + " " \
                            + courses['courseName'] + " " \
                            + courses ['moduleName'] + " " \
                            + courses['courseDesc'] + " " \
                            + courses['Outcome Description'] + " " \
                            + courses['competencyUnitDesc']
    courses = courses[['productId', 'Marketing Name', 'Description', 'jobFamily', 'competencyLevel']]
    courses['Description'] = courses['Description'].astype(str)
    courses['Description'] = courses['Description'].apply(preprocess)
    return courses

def tagcol_paragraph_embeddings_features(train_data: pd.DataFrame) -> list[TaggedDocument]:

    # Expects a dataframe with a 'Description' column
    train_data_values = train_data['Description'].values
    
    # Remember to use token.text to get the raw string, otherwise doc2vec cannot build vocabulary
    columns = [TaggedDocument([token.text for token in nlp(text) if token is not token.is_stop] , [i]) for i, text in enumerate(train_data_values)]
    
    return columns

def train_d2v_model(courses_dataset: pd.DataFrame) -> Doc2Vec:
    corpus = tagcol_paragraph_embeddings_features(courses_dataset)
    model = Doc2Vec(dm=0, vector_size=50, workers=multiprocessing.cpu_count(), min_count=2, epochs=100, hs=1, negative=0)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def course_suggestion_d2v(doc2vec_model: Doc2Vec, skills_gap_cand: dict, skills_gap_jobs: dict, courses_dataset: pd.DataFrame) -> dict:
    ans = {'job': {}, 'continuous learning': {}}

    for level in skills_gap_jobs:
        vector = doc2vec_model.infer_vector(skills_gap_jobs[level])
        res = doc2vec_model.dv.most_similar([vector], topn=20)
        course_unique = set()
        course_list = []
        for i, prob in res:
            if courses_dataset.loc[i, 'competencyLevel'][0] == str(level) and courses_dataset.loc[i, 'Marketing Name'] not in course_unique:
                course_unique.add(courses_dataset.loc[i, 'Marketing Name'])
                # course_list.append((courses_dataset.loc[i, 'productId'], courses_dataset.loc[i, 'Marketing Name'], courses_dataset.loc[i, 'competencyLevel']))
                course_list.append({
                    "Course ID": courses_dataset.loc[i, 'productId'],
                    "Course Name": courses_dataset.loc[i, 'Marketing Name'],
                    "Couse Level": courses_dataset.loc[i, 'competencyLevel']
                })
        ans['job'][level] = course_list[:5]

    for level in skills_gap_cand:
        vector = doc2vec_model.infer_vector(skills_gap_cand[level])
        res = doc2vec_model.dv.most_similar([vector], topn=20)
        course_unique = set()
        course_list = []
        for i, prob in res:
            if courses_dataset.loc[i, 'competencyLevel'][0] == str(level) and courses_dataset.loc[i, 'Marketing Name'] not in course_unique:
                course_unique.add(courses_dataset.loc[i, 'Marketing Name'])
                # course_list.append((courses_dataset.loc[i, 'productId'], courses_dataset.loc[i, 'Marketing Name'], courses_dataset.loc[i, 'competencyLevel']))
                course_list.append({
                    "Course ID": courses_dataset.loc[i, 'productId'],
                    "Course Name": courses_dataset.loc[i, 'Marketing Name'],
                    "Couse Level": courses_dataset.loc[i, 'competencyLevel']
                })
        ans['continuous learning'][level] = course_list[:5]

    return ans

def course_suggestion_spacy(spacy_model: spacy, skills_gap_cand: dict, skills_gap_jobs: dict, courses_dataset: pd.DataFrame) -> dict:
    ans = {'job': {}, 'continuous learning': {}}

    for level in skills_gap_jobs:
        skills_gap_text = " ".join(skills_gap_jobs[level])

        # get courses of same competency level
        df = courses_dataset[courses_dataset["competencyLevel"].str.contains(str(level))]
        df = df.copy()

        # get similarity score
        df["Similarity"] = df["Description"].apply(lambda x: spacy_model(skills_gap_text).similarity(spacy_model(str(x))))
        top_courses = df.nlargest(20,'Similarity',keep='all')
        course_unique = set()
        course_list = []
        for index, row in top_courses.iterrows():
            if row['Marketing Name'] not in course_unique:
                course_unique.add(row['Marketing Name'])
                # course_list.append((row['productId'], row['Marketing Name'], row['competencyLevel']))
                course_list.append({
                    "Course ID": row['productId'],
                    "Course Name": row['Marketing Name'],
                    "Couse Level": row['competencyLevel']
                })
        ans['job'][level] = course_list[:5]

    for level in skills_gap_cand:
        skills_gap_text = " ".join(skills_gap_cand[level])

        # get courses of same competency level
        df = courses_dataset[courses_dataset["competencyLevel"].str.contains(str(level))]
        df = df.copy()

        # get similarity score
        df["Similarity"] = df["Description"].apply(lambda x: spacy_model(skills_gap_text).similarity(spacy_model(str(x))))
        top_courses = df.nlargest(20,'Similarity',keep='all')
        course_unique = set()
        course_list = []
        for index, row in top_courses.iterrows():
            if row['Marketing Name'] not in course_unique:
                course_unique.add(row['Marketing Name'])
                # course_list.append((row['productId'], row['Marketing Name'], row['competencyLevel']))
                course_list.append({
                    "Course ID": row['productId'],
                    "Course Name": row['Marketing Name'],
                    "Couse Level": row['competencyLevel']
                })
        ans['continuous learning'][level] = course_list[:5]

    return ans

def final_course_suggestion_d2v(resume_text: str, peer_industry: str, job_name: str, skills_api: set, 
                                                                      resume_skills_required_pickle: dict, 
                                                                      job_skills_required_pickle: dict, 
                                                                      courses_dataset: pd.DataFrame, 
                                                                      doc2vec_model: Doc2Vec, 
                                                                      clean: bool=True):
    """
    This function returns list of courses corresponding to different competency levels for both job requirements and continuous learning.

    Parameters
    ----------
    resume_text : string
        The extracted text from a resume.
    peer_industry: string
        The industry that you want to compare yourself with. 
    job_name: string
        The particular role that applicants are applying to.
    skills_api : set
        A set that contains all the recognized skills. Created based on EMSI skills API.
    resume_skills_required_pickle: dictionary
        A dictionary that stores all the good-to-have skills and the corresponding competency levels for each industry.
    job_skills_required_pickle: dictionary
        A dictionary that stores all the job specific skills and the corresponding competency levels for each role.
    courses_dataset: pd.DataFrame
        The dataset that contains all the information of courses. Provided by Sambaash.
    doc2vec_model: doc2vec model
        The pretrained doc2vec model to compare the similarity between skills required and courses description.
    clean: boolean
        The boolean that indicates whether we shall clean the text.
      

    Returns
    -------
    dict (a nested dictionary that contains all the courses information)

    See Also
    --------
    final_course_suggestion_spacy : A similar method, but it used spacy pre-trained model instead of doc2vec model

    Examples
    --------
    >>> final_course_suggestion_d2v(resume_text, 'INFORMATION-TECHNOLOGY', 'Software Developers, Applications',
                                    skills_api, resume_skills_required_pickle, job_skills_required_pickle, courses_dataset, doc2vec_model, True)
    {'job': 
        {
            3: [(7203,'Innovation and Entrepreneurship Capstone','3 - Entrant Level'), ...],
            4: [(4602, 'Express Data Base Administrator', '4 - Specialist Level'), ...],
            5: [(12502, 'Cyber Security Management Capstone Project','5 - Expert Level'), ...]
        }
     'continuous learning':
        {
            3: [(7203,'Innovation and Entrepreneurship Capstone','3 - Entrant Level'), ...],
            4: [(4602, 'Express Data Base Administrator', '4 - Specialist Level'), ...],
            5: [(12502, 'Cyber Security Management Capstone Project','5 - Expert Level'), ...]
        }
    }
    """

    if peer_industry not in resume_skills_required_pickle:
        return 'Industry Not Found'
    if job_name not in job_skills_required_pickle:
        return 'Job Not Found'
    
    my_resume_skills = skills_experience_level_identification(resume_text, skills_api, clean)

    skills_gap_cand = skills_gap_by_level(skills_gap_identification(my_resume_skills, resume_skills_required_pickle[peer_industry]))
    skills_gap_jobs = skills_gap_by_level(skills_gap_identification(my_resume_skills, job_skills_required_pickle[job_name]))
    
    return course_suggestion_d2v(doc2vec_model, skills_gap_cand, skills_gap_jobs, courses_dataset)

def final_course_suggestion_spacy(resume_text: str, peer_industry: str, job_name: str, skills_api: set, 
                                                                      resume_skills_required_pickle: dict, 
                                                                      job_skills_required_pickle: dict, 
                                                                      courses_dataset: pd.DataFrame, 
                                                                      spacy_model: spacy.lang.en.English, 
                                                                      clean: bool=True):
    """
    This function returns list of courses corresponding to different competency levels for both job requirements and continuous learning.

    Parameters
    ----------
    resume_text : string
        The extracted text from a resume.
    peer_industry: string
        The industry that you want to compare yourself with.
    job_name: string
        The particular role that applicants are applying to.
    skills_api : set
        A set that contains all the recognized skills. Created based on EMSI skills API.
    resume_skills_required_pickle: dictionary
        A dictionary that stores all the good-to-have skills and the corresponding competency levels for each industry.
    job_skills_required_pickle: dictionary
        A dictionary that stores all the job specific skills and the corresponding competency levels for each role.
    courses_dataset: pd.DataFrame
        The dataset that contains all the information of courses. Provided by Sambaash.
    spacy_model: spacy pre-trained model
        The pretrained spacy model to compare the similarity between skills required and courses description.
    clean: boolean
        The boolean that indicates whether we shall clean the text.
      

    Returns
    -------
    dict (a nested dictionary that contains all the courses information)

    See Also
    --------
    final_course_suggestion_d2v: A similar method, but it used doc2vec model instead of spacy pre-trained model

    Examples
    --------
    >>> final_course_suggestion_spacy(resume_text, 'INFORMATION-TECHNOLOGY', 'Software Developers, Applications',
                                      skills_api, resume_skills_required_pickle, job_skills_required_pickle, courses_dataset, spacy_model, True)
    {'job': 
        {
            3: [(7203,'Innovation and Entrepreneurship Capstone','3 - Entrant Level'), ...],
            4: [(4602, 'Express Data Base Administrator', '4 - Specialist Level'), ...],
            5: [(12502, 'Cyber Security Management Capstone Project','5 - Expert Level'), ...]
        }
     'continuous learning':
        {
            3: [(7203,'Innovation and Entrepreneurship Capstone','3 - Entrant Level'), ...],
            4: [(4602, 'Express Data Base Administrator', '4 - Specialist Level'), ...],
            5: [(12502, 'Cyber Security Management Capstone Project','5 - Expert Level'), ...]
        }
    }
    """
    
    if peer_industry not in resume_skills_required_pickle:
        return 'Industry Not Found'
    if job_name not in job_skills_required_pickle:
        return 'Job Not Found'
    
    my_resume_skills = skills_experience_level_identification(resume_text, skills_api, clean)

    skills_gap_cand = skills_gap_by_level(skills_gap_identification(my_resume_skills, resume_skills_required_pickle[peer_industry]))
    skills_gap_jobs = skills_gap_by_level(skills_gap_identification(my_resume_skills, job_skills_required_pickle[job_name]))
    
    return course_suggestion_spacy(spacy_model, skills_gap_cand, skills_gap_jobs, courses_dataset)