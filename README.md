# csv2trec
Converts a CSV File Format to TREC compatible format (XML) , so that it can be used in search engines 
## Motivation
The motivation for this project came from the fact that to convert a CSV (TSV) file to a search engine compatible file. One may argue for a very long time about this that CSV can be imported. But a lot of other information are lost. In our case it was Amazon Comments Dataset. I wanted to perserve the important information such as category , the headline of the review and also the review content along with it's author.
## How does it work 
It takes TSV File (Currently Amazon Review comments Dataset) and then transforms into a XML format which is TREC/WSJ friendly
so that it can be read by a lot of search engine out there for indexing comments. 

To mass import .tsv files I've added a bash script (optional) , it just goes through the whole dir , scanning for .tsv file and then executes the script

## TO-DO
* Implement requirements so people can easily pip install and get it up and running
* Refactor some of it
