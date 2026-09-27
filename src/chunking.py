def chunk_text(text,chunk=300,overlap=50):
    chunks = []
    start = 0
    len_text = len(text)
    
    while (start<len_text): # tbb tk loop chale gy jbb tk text end tk na pohanch jaye
        end = start + chunk
        if end >= len_text:
            chunked = text[start:len_text].strip() # agr end prr aa gaye hain to jo hai bss us ko chunk bna do
            if chunked:
                chunks.append(chunked)
                break
        
        while end > start and text[end] != " ":
            end -=1 # word ko darmyan se cut nhi krna is lye peeche ale space tk jana hai 
        if end == start : # peeche koi bhi space nhi mila (aik hi lamba word hai)
            end = start + chunk
        
        chunked = text[start:end].strip()
        if chunked:
            chunks.append(chunked)
        
        start = end - overlap # next chunk peeche wale se overlap krna hai 
    return chunks
        