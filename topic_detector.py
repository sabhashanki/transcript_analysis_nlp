# !pip install sentence_transformers
# !pip install transformers
# Loading the model for embedding
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import math
from scipy.special import softmax
from pathlib import Path
import pickle

# TODO: code cleanup and redesigning is required

model1 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Get the summarized text


# List of main topics --> 17


topic_new_dict = {
'Law' : ["In modern politics, law and order is the approach focusing on harsher enforcement and penalties as ways to reduce crime.[1] Penalties for perpetrators of disorder may include longer terms of imprisonment", "The final judgement in the dispute was declared by the Supreme Court of India on 9 November 2019" , "Legal policy consists of principles the judges consider the law must uphold, such as: that law should serve the public interest, that it should be fair and just","A criminal charge is a formal accusation made by a governmental authority (usually a public prosecutor or the police) asserting that somebody has committed a crime","Court cases that involve disputes between people or businesses over money or some injury to personal rights are called “civil” cases", 'Human Rights',"A constitution is the aggregate of fundamental principles or established precedents that constitute the legal basis of a polity, organisation or other type of entity and commonly determine how that entity is to be governed"],
'Philosophy, thoughts and spirituality' : ["Philosophy is the systematized study of general and fundamental questions, such as those about existence, reason, knowledge, values, mind, and language","spirituality referred to a religious process of re-formation which aims to recover the original shape of man oriented at the image of God as exemplified by the founders and sacred texts of the religions of the world","Metaphysics is the branch of philosophy that studies the fundamental nature of reality; the first principles of being, identity and change, space and time, cause and effect, necessity and possibility","Ethics or moral philosophy is a branch[1] of philosophy that involves systematizing, defending, and recommending concepts of right and wrong behavior","A school of thought, or intellectual tradition, is the perspective of a group of people who share common characteristics of opinion or outlook of a philosophy,[1] discipline, belief, social movement, economics, cultural movement, or art movement"],
'Health and fitness' : ["Medicine is the science[1] and practice[2] of caring for a patient, managing the diagnosis, prognosis, prevention, treatment, palliation of their injury or disease, and promoting their health","Mindful Yoga[1] or Mindfulness Yoga[2] combines Buddhist-style mindfulness practice with yoga as exercise to provide a means of exercise that is also meditative and useful for reducing stress", "Hygiene is a series of practices performed to preserve health. Hygiene refers to conditions and practices that help to maintain health and prevent the spread of diseases","Health is a state of complete physical, mental and social well-being and not merely the absence of disease and infirmity", "Physical fitness is a state of health and well-being and, more specifically, the ability to perform aspects of sports, occupations and daily activities. Physical fitness is generally achieved through proper nutrition,[1] moderate-vigorous physical exercise,[2] and sufficient rest along with a formal recovery plan"],
'Governance and Politics' : ["Politics is the set of activities that are associated with making decisions in groups, or other forms of power relations among individuals, such as the distribution of resources or status. The branch of social science that studies politics and government is referred to as political science","Governance is the process of interactions through the laws, social norms, power (social and political) or language as structured in communication of an organized society[1] over a social system (family, social group, formal or informal organization, a territory under a jurisdication or across territories)", "An election is a formal group decision-making process by which a population chooses an individual or multiple individuals to hold public office","Public policy is an institutionalized proposal or a decided set of elements like laws, regulations, guidelines, and actions to solve or address relevant and real-world problems, guided by a conception and often implemented by programs. Public policy can be considered to be the sum of government direct and indirect activities and has been conceptualized in a variety of ways", "The Government of India has social welfare and social security schemes for India's citizens funded either by the central government, state government or concurrently. Schemes which are fully funded by the central government are referred to as central sector schemes (CS) while schemes mainly funded by the centre and implemented by the states are centrally sponsored schemes", "Public Administration (a form of governance) or Public Policy and Administration (an academic discipline) is the implementation of public policy, administration of government establishment (public governance), management of non-profit establishment (nonprofit governance), and also a subfield of political science taught in public policy schools that studies this implementation and prepares civil servants, especially those in administrative positions for working in the public sector", "Urban planning, also known as town planning, city planning, regional planning, or rural planning, is a technical and political process that is focused on the development and design of land use and the built environment, including air, water, and the infrastructure passing into and out of urban areas, such as transportation, communications, and distribution networks and their accessibility"],
'Arts' : ["Architecture is the art and technique of designing and building, as distinguished from the skills associated with construction.[3] It is both the process and the product of sketching, conceiving,[4] planning, designing, and constructing buildings or other structures","Poetry , also called verse,[note 1] is a form of literature that uses aesthetic and often rhythmic[1][2][3] qualities of language − such as phonaesthetics, sound symbolism, and metre − to evoke meanings in addition to, or in place of, a prosaic ostensible meaning", "Literature is any collection of written work, but it is also used more narrowly for writings specifically considered to be an art form, especially prose fiction, drama, and poetry", "History is also an academic discipline which uses narrative to describe, examine, question, and analyze past events, and investigate their patterns of cause and effect.","The arts are a very wide range of human practices of creative expression, storytelling and cultural participation", "Culture is an umbrella term which encompasses the collective social behavior, institutions, and norms found in human societies, which includes the collective knowledge, beliefs, arts, laws, customs, capabilities, and habits of the individuals in these groups", "The performing arts range from vocal and instrumental music, dance and theatre to pantomime, sung verse and beyond. They include numerous cultural expressions that reflect human creativity and that are also found, to some extent, in many other intangible cultural heritage domains"],
'Business, Finance and Economy' : ["Business is the practice of making one's living or making money by producing or buying and selling products (such as goods and services","Finance is the study and discipline of money, currency and capital assets. It is related to, but not synonymous with economics,which is the study of production, distribution, and consumption of money, assets, goods and services", "An economy is an area of the production, distribution and trade, as well as consumption of goods and services.","A stock market, equity market, or share market is the aggregation of buyers and sellers of stocks (also called shares), which represent ownership claims on businesses; these may include securities listed on a public stock exchange, as well as stock that is only traded privately, such as shares of private companies which are sold to investors through equity crowdfunding platforms. Investment is usually made with an investment strategy in mind.","Investment banking pertains to certain activities of a financial services company or a corporate division that consist in advisory-based financial transactions on behalf of individuals, corporations, and governments","Agriculture encompasses crop and livestock production, aquaculture, fisheries and forestry for food and non-food products","Industry, in economics and economic geography, refers to the production of an economic good or service within an economy.","Manufacturing is the creation or production of goods with the help of equipment, labor, machines, tools, and chemical or biological processing or formulation. It is the essence of the secondary sector of the economy","A startup or start-up is a company or project undertaken by an entrepreneur to seek, develop, and validate a scalable business model.[1][2] While entrepreneurship refers to all new businesses, including self-employment and businesses that never intend to become registered, startups refer to new businesses that intend to grow large beyond the solo founder.","E-commerce (electronic commerce) is the activity of electronically buying or selling of products on online services or over the Internet. E-commerce draws on technologies such as mobile commerce, electronic funds transfer, supply chain management, Internet marketing, online transaction processing, electronic data interchange (EDI), inventory management systems, and automated data collection systems","The Oil and Natural Gas Corporation (ONGC) is an Indian central public sector undertaking under the ownership of Ministry of Petroleum and Natural Gas,","Transport (in British English), or transportation (in American English), is the intentional movement of humans, animals, and goods from one location to another. Modes of transport include air, land (rail and road), water, cable, pipeline, and space","elecommunication is the transmission of information by various types of technologies over wire, radio, optical, or other electromagnetic systems","Employment is a relationship between two parties regulating the provision of paid labour services","A budget is a calculation plan, usually but not always financial, for a defined period, often one year or a month. A budget may include anticipated sales volumes and revenues, resource quantities including time, costs and expenses","A tax is a compulsory financial charge or some other type of levy imposed on a taxpayer (an individual or legal entity) by a governmental organization in order" ,"Trade involves the transfer of goods and services from one person or entity to another, often in exchange for money. Economists refer to a system or network that allows trade as a market", "International trade is the exchange of capital, goods, and services across international borders or territories[1] because there is a need or want of goods or services","Economics is the social science that studies the production, distribution, and consumption of goods and services. Economics focuses on the behaviour and interactions of economic agents and how economies work. Microeconomics analyzes what's viewed as basic elements in the economy, including individual agents and markets, their interactions, and the outcomes of interactions"],
'Entertainment' : ["Entertainment is a form of activity that holds the attention and interest of an audience or gives pleasure and delight","Celebrity is a condition of fame and broad public recognition of a person or group as a result of the attention given to them by mass media","Music is an art form consisting of sound and silence, expressed through time", "A film – also called a movie, motion picture, moving picture, picture, photoplay or (slang) flick – is a work of visual art that simulates experiences and otherwise communicates ideas, stories, perceptions, feelings, beauty, or atmosphere through the use of moving images. These images are generally accompanied by sound and, more rarely, other sensory stimulations.[1] The word cinema, short for cinematography, is often used to refer to filmmaking and the film industry, and to the art form that is the result of it"],
'Online Games' : ["A personal computer game, also known as computer game or PC game, is a type of video game played on a personal computer","PlayStation is a video gaming brand that consists of five home video game consoles as well as an online gaming service","A multiplayer video game is a video game in which more than one person can play in the same game environment at the same time"],
'Lifestyle' : ["A hobby is considered to be a regular activity that is done for enjoyment, typically during one's leisure time","A cuisine is a style of cooking characterized by distinctive ingredients, techniques and dishes, and usually associated with a specific culture or geographic region","the most common family type was one in which grandparents, parents, and children lived together as a single unit","The concept of interpersonal relationship involves social associations, connections, or affiliations between two or more people", "A pet, or companion animal, is an animal kept primarily for a person's company or entertainment rather than as a working animal, livestock, or a laboratory animal","Travel is the movement of people between distant geographical locations. Travel can be done by foot, bicycle, automobile, train, boat, bus, airplane, ship or other means, with or without luggage, and can be one way or round trip"],
'Science and Technology' : ["Space science encompasses all of the scientific disciplines that involve space exploration and study natural phenomena and physical bodies occurring in outer space, such as space medicine and astrobiology","Technological advancements have led to significant changes in society","Science is a systematic endeavor that builds and organizes knowledge in the form of testable explanations and predictions about the universe","Artificial intelligence (AI) is intelligence—perceiving, synthesizing, and inferring information—demonstrated by machines, as opposed to intelligence displayed by non-human animals and humans.","A cryptocurrency, crypto-currency, or crypto is a digital currency designed to work as a medium of exchange through a computer network","Robotics is an interdisciplinary branch of computer science and engineering.[1] Robotics involves the design, construction, operation, and use of robots."],
'Social Media' : ["Social media are interactive technologies that facilitate the creation and sharing of information, ideas, interests, and other forms of expression through virtual communities and networks", "When there is a buzz, conversations happening around a particular topic, person, or incident in the whole ecosystem, that means it's trending."," When a content piece suddenly breaks into the scene out of nowhere and gets a lot of eyeballs and engagement, it is a viral post"],
'Sports' : ["Cricket is a bat-and-ball game played between two teams of eleven players on a field at the centre of which is a 22-yard (20-metre) pitch with a wicket at each end, each comprising two bails balanced on three stumps.","Basketball is a team sport in which two teams, most commonly of five players each, opposing one another on a rectangular court, compete with the primary objective of shooting a basketball (approximately 9.4 inches (24 cm) in diameter) through the defender's hoop","Kabaddi is a contact team sport. Played between two teams of seven players, the objective of the game is for a single player on offence, referred to as a raider, to run into the opposing team's half of the court, touch out as many of their players and return to their own half of the court, all without being tackled by the defenders in 30 seconds","Football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal. Unqualified, the word football normally means the form of football that is the most popular where the word is used","Hockey is a term used to denote a family of various types of both summer and winter team sports which originated on either an outdoor field, sheet of ice, or dry floor such as in a gymnasium.","Mixed martial arts (MMA) is a full-contact combat sport based on striking, grappling and ground fighting, incorporating techniques from various combat sports from around the world.","Tennis is a racket sport that is played either individually against a single opponent (singles) or between two teams of two players each (doubles)"],
'World affairs' : ["Climate is the long-term weather pattern in a region, typically averaged over 30 years.[1][2] More rigorously, it is the mean and variability of meteorological variables over a time spanning from months to millions of years", "The natural environment or natural world encompasses all living and non-living things occurring naturally, meaning in this case not artificial","Geopolitics  is the study of the effects of Earth's geography (human and physical) on politics and international relations","A social issue is a problem that affects many people within a society. It is a group of common problems in present-day society and ones that many people strive to solve. It is often the consequence of factors extending beyond an individual's control. Social issues are the source of conflicting opinions on the grounds of what is perceived as morally correct or incorrect personal life or interpersonal social life decisions"]

}


# topic_new_dict = {
# 'Law' : ['Law and Order', 'Court Verdicts','Legal Policy','Criminal Charges','Civil cases', 'Human Rights','Constitution'],
# 'Philosophy, thoughts and spirituality' : ['Philosophy','Religion and Spirituality','Meta Physics','Moral values and Ethics','school of thoughts'],
# 'Health and fitness' : ['Medicine and Diagnosis','Yoga and Mindfulness', 'Health and Hygiene','Fitness and Excercise'],
# 'Governance and Politics' : ['Governance and Politics','Election and Political Parties','Public Policy and Welfare schemes', 'Public Administration'],
# 'Arts' : ['Building Architecture and Design','Poetry and Literature','History','Arts and Culture','Classical, Traditional Music and Dance', 'Traditional practices'],
# 'Business, Finance and Economy' : ['Business and Finance', 'Economy','Share Market','Investment and Banking','Agriculture','Industries and Manufacturing sector','Startup Companies','E-Commerce Sector','Oil and Gas sector','Transportation','Telecommunication','Employment and Human Resource','Budget and Taxation','Town Planning and Development','Trade, export and imports','Inflation'],
# 'Entertainment' : ['Entertainment Activities','Popular Celebrities','Music Albums, Podcasts and Concerts','Films, TV Shows and Movies','Film Shooting and Award Ceremonies'],
# 'Online Games' : ['Computer, PC Games','PlayStation Games','XBOX Games','Online Multiplayer Gaming'],
# 'Lifestyle' : ['Hobbies','Food and Drinks','Family and Relationship', 'Pets','Vlogging','Travelling and Exploring places'],
# 'Science and Technology' : ['Space Science','Technology','Science','Artificial Intelligence','Crytocurrency','Robotics','Technology development', 'Advance technologies','Automation'],
# 'Social Media' : ['Social Media Content', 'trending and viral posts','social network platforms'],
# 'Sports' : ['Cricket','Basketball','Kabaddi','Football','Hockey','Matial Arts','Gymnastics','Rugby','Wrestling','Tennis','Water Sports'],
# 'World affairs' : ['Climate and Environment','Geo-Politics','Social Issues']
# }

candidate_labels2 = ['Law', 'Philosophy, thoughts and spirituality', 'Health and fitness',
                    'Governance and Politics', 'Arts', 'Business, Finance and Economy',
                    'Entertainment', 'Online Games', 'Lifestyle', 'Science and Technology',
                    'Social Media', 'Sports', 'World affairs']

candidate_labels = ["Law - Court - Rights - Constitution", "Philosophical Thoughts - personal thoughts - religion", "Health and fitness",
                    "Governance and Politics", "Arts, Culture and literature", "Business, Finance and Economy",
                    "Entertainment - Film industry, Movies and TV Shows, Music Album and Concert - Movie Celebrities", "PC Gaming, Xbox, Playstation and Online Gaming", "Lifestyle, Hobbies, Food and Travel", "Science and Technology",
                    "Social Media", "Sports", "World affairs, Climate, Social Issues and Geo-Politics"]

# candidate_labels = ["Law - Court - Rights - Constitution", "Philosophical Thoughts - personal thoughts - religion", "Health and fitness",
#                     "Governance and Politics", "Arts, Culture and literature", "Business, Finance and Economy",
#                     "Entertainment - Films, Movies and TV Shows, Music Album and Concert - Movie Celebrities", "PC Gaming, Xbox, Playstation and Online Gaming", "Lifestyle, Hobbies, Food and Travel", "Science and Technology",
#                     "Social Media", "Sports", "World affairs, Climate, Social Issues and Geo-Politics"]


# candidate_labels = ['Law', 'Philosophy-thoughts-spirituality', 'Health-fitness', 
#                     'Governance-Politics', 'Arts','Business-Finance-Economy', 
#                     'Entertainment','Faiths','Games','Indian-Languages',
#                     'International-Languages','Lifestyle','Places','Science-Technology',
#                     'Social-Media','Sports','World-affairs']
topic_embeddings = {}


def get_embedding_cache(label):
    global topic_embeddings
    global model1

    if label not in topic_embeddings.keys():
        embed = model1.encode(label)
        topic_embeddings[label] = embed
    return topic_embeddings[label]


def get_topic_embeddings(topic_list):
    embedded_topic = []

    for i in range(len(topic_list)):
        embed = get_embedding_cache(topic_list[i])
        embedded_topic.append(embed)

    return embedded_topic


# >>> embedded_topic = get_topic_embeddings(candidate_labels)


def get_text_embeddings(text):
    embedded_text = []
    embed = model1.encode(text)
    embedded_text.append(embed)
    embedded_text = np.array(embedded_text)
    return embedded_text


# >>> embedded_text = get_text_embeddings(text_data)


# Function to get the cosine similarity
def cosine_similarity(v1, v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i];
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


def get_similarities_model1(embed_text, embed_topic):
    SCALE_SCORE = 5
    scored = []
    for i in range(len(embed_topic)):
        for j in range(len(embed_text)):
            score = np.round(cosine_similarity((embed_text[j]), (embed_topic[i])), 3)
            scored.append(score * SCALE_SCORE)
    print(f"Before: {scored}")
    scored = softmax(np.array(scored))
    print(f"After: {scored}")
    return scored


# >>> score = get_similarities_model1(embedded_text, embedded_topic)


# >>> get_highest_scored_topics(score)

def get_topics(text):
    embedded_text = get_text_embeddings(text)
    embedded_topic = get_topic_embeddings(candidate_labels)
    score = get_similarities_model1(embedded_text, embedded_topic)
    topic_score_list = sorted(zip(candidate_labels2, score), key=lambda x: x[1], reverse=True)
    print(f"score: {topic_score_list}")
    return topic_score_list[:3]

# # Segmentation fault
# def get_topics(text):
#     embedded_text = get_text_embeddings(text)
#     final_score = []
#     final_labels = []
#     for topic, labels in topic_new_dict.items():
#         topic_wise_score = []
#         embedded_topic = get_topic_embeddings(labels)
#         score = get_similarities_model1(embedded_text, embedded_topic)
#         final_score.append(max(score))
#         label_highscore = labels[score.index(max(score))]
#         final_labels.append(label_highscore)
#     topic_score_list = sorted(zip(final_labels, final_score), key=lambda x: x[1], reverse=True)
#     print(f"score: {topic_score_list}")
#     return topic_score_list


# Working subtopic phrases
# def get_topics(text):
#     embedded_text = get_text_embeddings(text)
#     final_score = []
#     final_labels = []
#     for topic, labels in topic_new_dict.items():
#         embedded_topic = get_topic_embeddings(labels)
#         score = get_similarities_model1(embedded_text, embedded_topic)
#         final_score.append(score.max())
#         # final_labels.append(labels[np.argmax(score)])
#     topic_score_list = sorted(zip(list(topic_new_dict), final_score), key=lambda x: x[1], reverse=True)
#     print(f"score: {topic_score_list}")
#     return topic_score_list


def get_topics_top(top_list, text):
    embedded_text = get_text_embeddings(text)
    final_score = []
    final_label = []
    for tup in top_list:
        labels = topic_new_dict[tup[0]]
        embedded_topic = get_topic_embeddings(labels)
        score = get_similarities_model1(embedded_text, embedded_topic)
        final_score.append(score.max())
        final_label.append(labels[np.argmax(score)])
    topic_score_list = sorted(zip(final_label, final_score), key=lambda x: x[1], reverse=True)
    return topic_score_list




def get_topic_gpt3(fname):
    global candidate_labels
    SCALE_SCORE = 50
    score = []
    summary_embed = get_summary_embedding(fname)
    for lbl in candidate_labels:
        s = similarity(get_label_embedding(lbl[:3]), summary_embed)
        score.append(s * SCALE_SCORE)
    score = softmax(np.array(score))
    topic_score_list = sorted(zip(candidate_labels2, score), key=lambda x: x[1], reverse=True)
    return topic_score_list


def similarity(vA, vB):
    return np.dot(vA, vB) / (np.linalg.norm(vA) * np.linalg.norm(vB))


def get_label_embedding(lbl):
    output_embedding = None
    path_to_pkl = Path("openai/embeddings/topic_embeddings").joinpath(f"{lbl}.pkl")
    if path_to_pkl.exists():
        with open(path_to_pkl, 'rb') as pkl_file:
            output_embedding = np.array(pickle.load(pkl_file))
            print(f"Shape of label: {output_embedding.shape}")
    else:
        assert (False)
    return output_embedding


def get_summary_embedding(fname):
    summary_embedding = None
    path_to_pkl = Path("openai/embeddings/summary_embeddings").joinpath(f"{fname}.pkl")
    if path_to_pkl.exists():
        with open(path_to_pkl, 'rb') as pkl_file:
            summary_embedding = np.array(pickle.load(pkl_file))
            print(f"Shape of summary: {summary_embedding.shape}")
    else:
        assert (False)
    return summary_embedding
