from apiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import json
import csv

def get_id(url):
    u_pars = urlparse(url)
    quer_v = parse_qs(u_pars.query).get('v')
    if quer_v:
        return quer_v[0]
    pth = u_pars.path.split('/')
    if pth:
        return pth[-1]

def build_service(filename):
    with open(filename) as f:
        key = f.readline()

    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    return build(YOUTUBE_API_SERVICE_NAME,
                 YOUTUBE_API_VERSION,
                 developerKey=key)


def get_comments(part='id, snippet, replies',
                 maxResults=100,
                 textFormat='plainText',
                 order='time',
                 videoId="https://www.youtube.com/watch?v=hD1YtmKXNb4",
                 csv_filename="comments"):
    
    # create empty lists to store information
    comments, commentsId, repliesCount, isReplies, authorNames, authorIds = [], [], [], [], [], []
    file = open(f'{csv_filename}.csv','w', newline='', encoding="utf-8")
    
    with file:
        header = ["Comments", "Comment ID", "Reply Count", "Is Reply", "Author", "AuthorID"]
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()

        #build our service from api key
        service = build_service('Api_Key.txt')

        #make an API call using our service
        response = service.commentThreads().list(
            part=part,
            maxResults=maxResults,
            textFormat=textFormat,
            order=order,
            videoId=get_id(videoId)
        ).execute()

        while response: #this loop will continue to run until you max out your quota
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comment_id = item['snippet']['topLevelComment']['id']
                reply_count = item['snippet']['totalReplyCount']
                isReply = False
                authorName = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                if 'authorChannelId' in item['snippet']['topLevelComment']['snippet']:
                    authorId =  item['snippet']['topLevelComment']['snippet']['authorChannelId']['value']
                else:
                    authorId = None

                #append to lists
                comments.append(comment)
                commentsId.append(comment_id)
                repliesCount.append(reply_count)
                isReplies.append(isReply)
                authorNames.append(authorName)
                authorIds.append(authorId)



                    
                writer.writerow({'Comments' : comment,
                                     'Comment ID' : comment_id,
                                     'Reply Count' : reply_count,
                                     'Is Reply' : isReply,
                                     'Author' : authorName,
                                     'AuthorID' : authorId})
                
                if reply_count > 0 and 'replies' in item:
                    for reply in item['replies']['comments']:
                        reply_comment = reply['snippet']['textDisplay']
                        reply_comment_id = reply['id']
                        reply_reply_count = 0
                        isReply = True
                        reply_author_name = reply['snippet']['authorDisplayName']
                        if 'authorChannelId' in reply['snippet']:
                            reply_author_id = reply['snippet']['authorChannelId']['value']
                        else:
                            reply_author_id = None

                        #append to lists
                        comments.append(reply_comment)
                        commentsId.append(reply_comment_id)
                        repliesCount.append(reply_reply_count)
                        isReplies.append(isReply)
                        authorNames.append(reply_author_name)
                        authorIds.append(reply_author_id)

                        #write line by line
                        writer.writerow({'Comments' : reply_comment,
                                             'Comment ID' : reply_comment_id,
                                             'Reply Count' : reply_reply_count,
                                             'Is Reply' : isReply,
                                             'Author' : reply_author_name,
                                             'AuthorID' : reply_author_id})
                        


            # check for nextPageToken, and if it exists, set response equal to the Json response
            if 'nextPageToken' in response:
                response = service.commentThreads().list(
                    part=part,
                    maxResults=maxResults,
                    textFormat=textFormat,
                    order=order,
                    videoId=get_id(videoId),
                    pageToken=response['nextPageToken']
                ).execute()
            else:
                break
    
    # return our data of interest

    return {
        'Comments' : comments,
        'Comment ID' : commentsId,
        'Reply Count' : repliesCount,
        'isReplies' : isReplies,
        'Author Names' : authorNames,
        'Author Ids' : authorIds
    }


get_comments()