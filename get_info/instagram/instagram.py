import requests

# URL for the Instagram hashtag API
url = 'https://www.instagram.com/explore/tags/summer/?__a=1'

# Sending a GET request to fetch the data
response = requests.get(url)

if response.status_code == 200:
    print(response.text)
    
    # data = response.json()  # Parse the response JSON

    # # Looping through the first 6 media posts
    # for i in range(6):
    #     media_node = data['graphql']['hashtag']['edge_hashtag_to_media']['edges'][i]['node']
        
    #     # Extracting the thumbnail URL
    #     thumbnail_url = media_node['thumbnail_resources'][2]['src']
    #     print(f'<img src="{thumbnail_url}">')  # You can replace print with any desired method to process the URL
else:
    print(f"Error: {response.status_code}")  # Handle the case where the request fails


"""
$.get('https://www.instagram.com/explore/tags/summer/?__a=1', function (data, status) {
for(var i = 0; i < 6; i++) {
    var $this = data.graphql.hashtag.edge_hashtag_to_media.edges[i].node;
    $('#container').append('<img src="'+  $this.thumbnail_resources[2].src +'">');
}
});
"""