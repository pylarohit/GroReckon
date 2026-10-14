import urllib.request
import json
import re

url = "https://www.youtube.com/watch?v=w6l_-iElGDs"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
try:
    with urllib.request.urlopen(req) as response:
        html = response.read().decode("utf-8", errors="ignore")
    
    # Extract title
    title_match = re.search(r'<title>(.*?)</title>', html)
    print("PAGE TITLE:", title_match.group(1) if title_match else "N/A")
    
    # Extract short description
    desc_match = re.search(r'"shortDescription":"(.*?)"(?:,"isCrawlable"|,"thumbnail")', html)
    if desc_match:
        raw_desc = desc_match.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\/', '/')
        with open("video_description.txt", "w", encoding="utf-8") as f:
            f.write(raw_desc)
        print("DESCRIPTION SAVED TO video_description.txt")
    
    # Check captions
    caption_tracks = re.findall(r'"captionTracks":\[(.*?)\]', html)
    if caption_tracks:
        print("\nCaption tracks found:")
        # Find baseUrl
        urls = re.findall(r'"baseUrl":"(.*?)"', caption_tracks[0])
        for u in urls:
            clean_url = u.replace('\\u0026', '&')
            print("Caption URL:", clean_url)
            try:
                cap_req = urllib.request.Request(clean_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(cap_req) as cap_res:
                    cap_xml = cap_res.read().decode("utf-8", errors="ignore")
                    # strip xml tags to get text
                    lines = re.findall(r'<text[^>]*>(.*?)</text>', cap_xml)
                    clean_lines = [re.sub(r'&amp;#39;', "'", re.sub(r'&quot;', '"', l)) for l in lines]
                    print("\n--- TRANSCRIPT EXCERPT (first 40 lines) ---")
                    for line in clean_lines[:40]:
                        print(line)
                    with open("transcript.txt", "w", encoding="utf-8") as tf:
                        tf.write("\n".join(clean_lines))
                    print(f"Full transcript written to transcript.txt ({len(clean_lines)} lines)")
            except Exception as e:
                print("Error reading caption track:", e)
            break
    else:
        print("No captionTracks found in HTML")
except Exception as e:
    print("Error:", e)
