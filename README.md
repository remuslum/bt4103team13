
# Skills-Engine

## What is the problem? 

Given a candidate resume and a query of desired role, identify the skill gap between the candidate and the desired role, then generate a unique learning path for the candidate.  

## What is this project for?

This project is a testing playground that was used to record all progress for the project and how to solve the problems accordingly. 

## Instructions to run the notebook

Place all the files in `./app/datasets/` and `./app/pkl_files/` in the same directory level as `./app/Skills Engine.ipynb`. Sections 1 to 5 are the walkthrough sections while Section 6 is the final compilation of all the functions.

## Process

### Step 1 Skills Extraction

In order to efficiently extract skills out from the resume, use free online **EMSI Skills** api to get the list of available skills. 

- **EMSI Skills API**

    Example: calling the API endpoint using Python. The detailed documentation is given [Here]([Link text Here](https://link-url-here.org))

    ```
    url = "https://auth.emsicloud.com/connect/token"

    payload = "client_id=<CLIENT_ID>&client_secret=<CLIENT_SECRET>&grant_type=client_credentials&scope=emsi_open"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    response = requests.request("POST", url, data=payload, headers=headers)
    ```

- **Skills Extraction from Resume**

    Example: extract all the skills from a given resume. The output will be in a Python set.

    ```
    extract_skills(resume_text=resume_string, skills_api=skills_api, clean=True)

    # Output:
    # {
    #     'ada',
    #     'business administration',
    #     'coaching',
    #     'conflict resolution',
    #     'environmental quality',
    #     'labor law',
    #     'management',
    #     'mediation',
    #     'organizational development',
    #     'policy development',
    #     ...
    # }
    ```

### Step 2 Identification of Year of Experience

The output shown above is implicitly indicating that all the skills have the same experience level. This is undesirable as there is much more information in the resume that can be used to weigh the skills differently, e.g. project period. 

In order to weigh each skill differently based on the project period, we assumed that:
- All the experience levels are using `year(s)` as unit.
- All the skills extracted from the resume have at least 1 year of experience.
- The project description shall come after the project period.
- The year of experience is computed by taking the ceiling of the year difference, i.e. $\text{difference in time} // \text{365 days} + 1$
- When there are multiple projects of different periods having the same skill, the experience level will be computed by $max(\text{years of project 1}, \text{years of project 2}...)$.
- The title of a section is an **uppercased Noun**

We will first break down the resume text into sections using *title*, for each section, we then break down the paragraph into subsections using *date range*

Example: extract all the sections and subsections out using title and date range as guidance from a resume.

```
experience_tagging(date_list)

# Output:
# key: year of experience
# value: list of tuples, where each tuple is (section start index, section end index)

# {10: [(981, 1066)], 4: [(1996, 3608)], 3: [(3635, -1)]}
```

### Step 3 Experience Level Tagging

We looped through each section and subsection to extract all the skills out, then tag all these skills with the corresponding years of experience. For all the other skills that are not found within any sections, we just tag the year of experience to be 1. 

Example: extract all the skills with corresponding years from a resume. The output is in a Python
dictionary.

```
skills_experience_level_identification(resume_string)

# Output:
# {
#   'reduction': 4,
#   'environmental quality': 4,
#   'teamwork': 4,
#   'management': 4,
#   'poultry': 4,
#   'strong work ethic': 4,
#   'business administration': 3,
#   'coaching': 3,
#   'conflict resolution': 3,
#   'workplace safety': 3,
#   'mediation': 3,
#   'organizational development': 3,
#   'session': 3,
#   'ada': 3,
#   'labor law': 1,
#   'policy development': 1
# }
```

### Step 4: Identification of Significant Skills

There are 2 sets of documents that significant skills can be identified from, job-description documents and resume documents.
- From job-description documents, we extract the required skills to meet current industry standards.
- From resume documents, we extract the required skills to standout among peers in the same industry. Assumption: all candidates applying for the specific role **must** be qualified, i.e. they have all the skills required for the particular role. 

Aggregation is performed on all documents that belong to the same category to find out the frequency of documents that have each specified skill. High frequency indicates that they are the common skills required for the particular role. 

We then set the significance level $\alpha$ as .95, hence all the skills below 95 percentile will be treated as not so significant. The remaining frequencies are scaled to the range of [0, 1] to help us decide on the competency level to be given to each skill.

Example: We use the **percentile** method to set the threshold for classifying the skills significance level. We save all these remaining skills with the corresponding competency level required into a dictionary.

```   
# Output: where the value is the competency level based on the client's business settings.
# {'ADVOCATE': 
#  {'active listening': 3,
#   'administrative support': 3,
#   'advocacy': 3,
#   'agenda': 3,
#   'auditing': 3,
#   'authorization': 3,
#   'banking': 3,
#   'billing': 3,
#   'budgeting': 3,
#   'business administration': 3,
#   'business development': 3,
#   ...
#  }, 
#  ...
# }
```

### Step 5: Identification of Skill Gap

With the significant skills identified for each industry, we can now make comparison between your skillset and the skillset required for a particular industry. We can do that by first mapping the years of experience into **pre-defined competency levels: 3 for entrant level, 4 for specialist level and 5 for expert level.** Then a simple intersection is performed to get the skills gap, i.e. $\text{Required Level 5} - \text{User Level 3} = \text{Skills Gap [4,5]}$.

Example: The output will be a dictionary with skills as the `keys` and the skills gap as the `values`

```
skill_gap_identification_peers(skills[31605080], skills_required['AVIATION'])

# Output:
# {'aircraft maintenance': [3],
#  'aviation': [4, 5],
#  'b': [3],
#  'business administration': [3],
#  'c': [3],
#  'construction': [3],
#  'critical thinking': [3],
#  'drawing': [3],
#  'filing': [3],
#   ...
#  'supervision': [3],
#  'test equipment': [3],
#  'time management': [3],
#  'track': [3, 4],
#  'tracking': [3]
# }
```

### Step 6: Generating The Learning Path

Once we successfully find out the skills gap between an applicant and the job and peer standards, we can suggest the applicant to **job** and **continuous learning** programmes which help the applicant to fulfill the skills gap and even outperforming their peers in the specific industry. 

Example: We infer skills required in a vector form using Doc2Vec and find the top 5 unique most similar courses

```
# Output:
# {'job': 
#        {
#            3: [(7203,'Innovation and Entrepreneurship Capstone','3 - Entrant Level'), ...],
#            4: [(4602, 'Express Data Base Administrator', '4 - Specialist Level'), ...],
#            5: [(12502, 'Cyber Security Management Capstone Project','5 - Expert Level'), ...]
#        },
#  'continuous learning':
#        {
#            3: [(7203,'Innovation and Entrepreneurship Capstone','3 - Entrant Level'), ...],
#            4: [(4602, 'Express Data Base Administrator', '4 - Specialist Level'), ...],
#            5: [(12502, 'Cyber Security Management Capstone Project','5 - Expert Level'), ...]
#        }
# }
```

## Final Function

The Final Function is a single function that takes in a user's resume and desired industry/job name to output a learning pathway to fulfill their skills gap idenitified. Refer to **Section 6** of `Skills Engine.ipynb` for the following information.

**6.1** contains the Python Libraries required to run this final function.

**6.2** contains the Helper Functions that make up this final function.

**6.3** contains the Pre-Training Codes that will create the pickle files required for the final function to run. Can be used to reproduce this resources whenever there is any change to their sources.
- **`skills_api.pkl`** used in skills extraction, created from `all_skills_emsi.xlsx`.
- **`resume_skills_required.pkl`** used in skills gap identification, created from `Resume.csv`.
- **`job_skills_required.pkl`** used in skills gap identification, created from `Job Description.csv`.
- **`d2v_model.pkl`** used in learning pathway generation, trained from `Courses.xlsx`.

**6.4** contains the Final Function with detailed comments.
```
final_course_suggestion_d2v(resume_text, 'INFORMATION-TECHNOLOGY', 'Software Developers, Applications',
                            skills_api, resume_skills_required_pickle, job_skills_required_pickle, 
                            courses_dataset, doc2vec_model, True)
```

## How to Build and Tag a Docker Image?

Due to the time remaining for devlopment is limited and we need to start compiling and documenting all the source codes, the current limitation is:
 - we do not have the access to directly call PUT request to save a pdf file to S3 bucket via API Gateway
 - we do not have the access to read pdf files from S3 bucket via Lambda (errorMessage: *when calling GetObject operation: Access Denied*)

Hence, we decided to use GET request with the key to be `doc` and the value to be *resume_string*. This can be easily modified to only take the pdf filename once the access to read files from S3 bucket is given. 

Follow the instructions below to build the Docker image:

 1. Make sure that you have the latest version of the AWS CLI and Docker installed.
 2. Type `aws configure` in cmd and specify the correct Access Key ID and Secret Access Key. They can be found from AWS IAM.
 3. Type `cd "<path_to_folder>/bt4103team13/app/SkillsExtract Docker"`
 4. Retrieve an authentication token and authenticate your Docker client to your registry. Further details can be referred from AWS ECR repository -> *view push commands*
 5. The current directory structure shall be:
```
SkillsExtract Docker
|__ /app
|__ Dockerfile
|__ requirements.txt
```
 6. Type `docker build -t skills-engine .`
 7. Further details can be referred from AWS ECR repository -> *view push commands*

The current Docker Image only includes the function to extract skills and corresponding years of experience for the demonstration purpose. To include more `lambda_handler` to show skill gap and recommended courses, follow the instructions below:  

 1. Replace `./app/SkillsExtract Docker/requirements.txt` with `./app/requirements.txt`
 2. Replace `./app/SkillsExtract Docker/app/functions.py` with `./app/functions.py`
 3. Create other `.py` files that contain the `lambda_handler` in the same directory level as `./app/SkillsExtract Docker/app/lambda_function.py`
 4. Add all the other necessary `.pkl` files to the current directory level from `./app/pkl_files`
 5. Modify `./app/SkillsExtract Docker/Dockerfile` correspondingly to be able to call different `lambda_handlers`






