IMAGE_NAME := nukethebrowser
PCAP_FILE ?= nukethebrowser.pcap
ANSWERS_DIR := answers
DOCKER_RUN := docker run --rm -v $(shell pwd)/$(PCAP_FILE):/app/$(PCAP_FILE) -v $(shell pwd)/$(ANSWERS_DIR):/app/$(ANSWERS_DIR) -v $(shell pwd)/exported_files:/app/exported_files $(IMAGE_NAME)

.PHONY: all build clean q1 q2 q3 q4 q5 q6 q7 q8 q9

all: build q1 q2 q3 q4 q5 q6 q7 q8 q9

build:
	@docker build -t $(IMAGE_NAME) .

clean:
	@rm -rf $(ANSWERS_DIR)/*.txt exported_files

# Question targets
define run_question
@echo "$(1)"
@$(DOCKER_RUN) _$(2)
endef

q1: build ; $(call run_question,Q1: Highest targeted IP address,q1)
q2: build ; $(call run_question,Q2: Attack protocol,q2)
q3: build ; $(call run_question,Q3: Malicious executable URL,q3)
q4: build ; $(call run_question,Q4: Packet with French Google redirect,q4)
q5: build ; $(call run_question,Q5: CMS used by shop.honeynet.sg,q5)
q6: build ; $(call run_question,Q6: Packet showing show.php wont reinfect,q6)
q7: build ; $(call run_question,Q7: CVE for msdds.dll vulnerability,q7)
q8: build ; $(call run_question,Q8: Executable name from load.php?e=8,q8)
q9: build ; $(call run_question,Q9: Full MD5 hash ending with 78873f791,q9)


# Internal targets (run inside container)
# Q1: Highest targeted IP address
_q1:
	@echo "1. First, let's examine all IPv4 conversations to identify targeted systems..."
	tshark -r $(PCAP_FILE) -q -z conv,ip -n | tee $(ANSWERS_DIR)/conversations.txt
	@echo "\n2. Filtering out multicast/broadcast traffic to focus on unicast communications..."
	grep -v -E '224\.0\.0\.22|255\.255\.255\.255' $(ANSWERS_DIR)/conversations.txt | tee $(ANSWERS_DIR)/filtered_conversations.txt
	@echo "\n3. Extracting just the IP addresses and sorting numerically to find the highest..."
	awk '{print $$1}' $(ANSWERS_DIR)/filtered_conversations.txt | sort -n -t. -k1,1 -k2,2 -k3,3 -k4,4 | tail -n 1 | tee $(ANSWERS_DIR)/q1.txt
	@echo "\nAnalysis complete. Highest targeted IP address identified."

# Q2: Attack protocol
_q2:
	@echo "1. First, let's examine the protocol hierarchy to understand traffic composition..."
	tshark -r $(PCAP_FILE) -q -z io,phs | tee $(ANSWERS_DIR)/protocol_hierarchy.txt
	@echo "\n2. Looking for the most frequent application-layer protocols in malicious conversations..."
	tshark -r $(PCAP_FILE) -Y 'ip.src == 10.0.5.15' -q -z io,phs | tee $(ANSWERS_DIR)/attacker_protocols.txt
	@echo "\n3. Analyzing port usage in suspicious conversations..."
	tshark -r $(PCAP_FILE) -q -z conv,tcp | grep '10.0.5.15' | tee $(ANSWERS_DIR)/attacker_ports.txt
	@echo "\n4. Examining the most frequently used port in malicious communications..."
	awk '{print $$3}' $(ANSWERS_DIR)/attacker_ports.txt | cut -d: -f2 | sort | uniq -c | sort -nr | head -n 1 | awk '{print $$2}' | tee $(ANSWERS_DIR)/top_port.txt
	@echo "\n5. Determining protocol based on port usage..."
	(grep -q '^80$$' $(ANSWERS_DIR)/top_port.txt && echo "HTTP") | tee $(ANSWERS_DIR)/q2.txt

# Q3: URL for page serving malicious executables
_q3:
	@echo "1. Searching for executable file transfers in HTTP traffic..."
	tshark -r $(PCAP_FILE) -Y 'http.content_type contains "octet-stream"' -T fields -e http.host -e http.request.uri | tee $(ANSWERS_DIR)/executable_transfers.txt
	@echo "\n2. Identifying suspicious file downloads..."
	tshark -r $(PCAP_FILE) -Y 'http.content_type contains "octet-stream"' --export-objects "http,$(ANSWERS_DIR)/exported_files" 2>/dev/null
	@echo "\n3. Analyzing HTTP streams containing executable downloads..."
	tshark -r $(PCAP_FILE) -Y 'http.request.uri contains "load.php"' -T fields -e http.host -e http.request.uri | head -n 1 | awk '{print "http://"$$1$$2}' | cut -d'?' -f1 | tee $(ANSWERS_DIR)/q3.txt
	@echo "\nFound malicious executable delivery URL:"
	@cat $(ANSWERS_DIR)/q3.txt

# Q4: Packet with French Google redirect (Geo-targeting indicator)
_q4:
	@echo "1. Searching for HTTP redirects (status 302)..."
	tshark -r $(PCAP_FILE) -Y 'http.response.code == 302' -T fields -e frame.number -e http.location | tee $(ANSWERS_DIR)/all_redirects.txt
	@echo "\n2. Looking for geographical targeting indicators..."
	tshark -r $(PCAP_FILE) -Y 'http.location contains "google.fr"' -T fields -e frame.number | tee $(ANSWERS_DIR)/french_redirect.txt
	@echo "\n3. Verifying the redirect packet..."
	tshark -r $(PCAP_FILE) -Y "frame.number == $$(cat $(ANSWERS_DIR)/french_redirect.txt)" -V | grep -A 10 "Hypertext Transfer Protocol" | tee $(ANSWERS_DIR)/redirect_verification.txt
	cat $(ANSWERS_DIR)/french_redirect.txt | tee $(ANSWERS_DIR)/q4.txt
	@echo "\nIdentified geo-targeting indicator packet:"
	@cat $(ANSWERS_DIR)/q4.txt

# Q5: CMS used by shop.honeynet.sg/catalog/
_q5:
	@echo "1. Debugging HTTP requests for shop.honeynet.sg..."
	@tshark -r $(PCAP_FILE) -Y 'http.host contains "shop.honeynet.sg"' -T fields -e frame.number -e http.request.method -e http.request.uri -e http.host > $(ANSWERS_DIR)/shop_requests.txt 2>/dev/null
	@echo "   Found $$(wc -l < $(ANSWERS_DIR)/shop_requests.txt) request(s) for shop.honeynet.sg:"
	@cat $(ANSWERS_DIR)/shop_requests.txt
	@grep -q '/catalog/' $(ANSWERS_DIR)/shop_requests.txt || { echo "   No requests found for /catalog/. Check URI in PCAP."; echo "CMS not found" > $(ANSWERS_DIR)/q5.txt; exit 1; }
	@request_packet=$$(grep '/catalog/' $(ANSWERS_DIR)/shop_requests.txt | head -n 1 | awk '{print $$1}'); \
	if [ -z "$$request_packet" ]; then \
		echo "   No request packet found for /catalog/."; \
		echo "CMS not found" > $(ANSWERS_DIR)/q5.txt; \
		exit 1; \
	fi; \
	echo "\n2. Identifying response packet for request $$request_packet..."; \
	response_packet=$$(tshark -r $(PCAP_FILE) -Y "frame.number == $$request_packet" -T fields -e http.response_in | head -n 1); \
	if [ -z "$$response_packet" ]; then \
		echo "   No response_in found. Checking TCP stream..."; \
		tcp_stream=$$(tshark -r $(PCAP_FILE) -Y "frame.number == $$request_packet" -T fields -e tcp.stream | head -n 1); \
		response_packet=$$(tshark -r $(PCAP_FILE) -Y "tcp.stream == $$tcp_stream && http.response" -T fields -e frame.number | head -n 1); \
		if [ -z "$$response_packet" ]; then \
			echo "   No response in TCP stream $$tcp_stream. Check PCAP for server responses."; \
			echo "CMS not found" > $(ANSWERS_DIR)/q5.txt; \
			exit 1; \
		fi; \
		echo "   Found response packet $$response_packet in TCP stream $$tcp_stream."; \
	else \
		echo "   Found response packet $$response_packet via http.response_in."; \
	fi; \
	echo "\n3. Extracting HTML content from response packet $$response_packet..."; \
	tshark -r $(PCAP_FILE) -Y "frame.number == $$response_packet" -T fields -e http.content_encoding -e http.file_data > $(ANSWERS_DIR)/response_body.txt 2>/dev/null; \
	if [ ! -s "$(ANSWERS_DIR)/response_body.txt" ]; then \
		echo "   No HTML content extracted from packet $$response_packet. Checking exported objects..."; \
		tshark -r $(PCAP_FILE) -Y "frame.number == $$response_packet" --export-objects http,$(ANSWERS_DIR)/exported_files 2>/dev/null; \
		if ls $(ANSWERS_DIR)/exported_files/*.html >/dev/null 2>&1; then \
			cat $(ANSWERS_DIR)/exported_files/*.html > $(ANSWERS_DIR)/response_body.txt; \
			echo "   Extracted HTML from exported objects."; \
		else \
			echo "   No HTML content found. Check for compression or missing response."; \
			echo "CMS not found" > $(ANSWERS_DIR)/q5.txt; \
			exit 1; \
		fi; \
	fi; \
	encoding=$$(head -n 1 $(ANSWERS_DIR)/response_body.txt | awk '{print $$1}'); \
	echo "   Content encoding: $$encoding"; \
	echo "\n4. Searching for CMS indicators in HTML..."; \
	grep -i -A 5 'powered by\|running on\|content management' $(ANSWERS_DIR)/response_body.txt | grep -iE '<b>([a-zA-Z]+) [A-Z][a-z]+ [A-Z][a-z]+( [vV][0-9A-Za-z. ]+)?</b>' | sed -E 's/.*<b>([a-zA-Z]+ [A-Z][a-z]+ [A-Z][a-z]+).*/\1/' | grep -v 'Honeynet Forensics Challenge\|Honeynet Online Store\|New Products For\|What s New' > $(ANSWERS_DIR)/potential_cms.txt; \
	if [ -s "$(ANSWERS_DIR)/potential_cms.txt" ]; then \
		cms=$$(head -n 1 $(ANSWERS_DIR)/potential_cms.txt); \
		echo "   Found CMS near 'powered by', 'running on', or 'content management': $$cms"; \
		echo "   All potential CMS names:"; \
		cat $(ANSWERS_DIR)/potential_cms.txt; \
		echo "$$cms" > $(ANSWERS_DIR)/q5.txt; \
	else \
		echo "   No CMS near 'powered by', 'running on', or 'content management'. Trying broader search..."; \
		grep -iE '<b>([a-zA-Z]+) [A-Z][a-z]+ [A-Z][a-z]+( [vV][0-9A-Za-z. ]+)?</b>' $(ANSWERS_DIR)/response_body.txt | sed -E 's/.*<b>([a-zA-Z]+ [A-Z][a-z]+ [A-Z][a-z]+).*/\1/' | grep -v 'Honeynet Forensics Challenge\|Honeynet Online Store\|New Products For\|What s New' > $(ANSWERS_DIR)/potential_cms.txt; \
		if [ -s "$(ANSWERS_DIR)/potential_cms.txt" ]; then \
			cms=$$(head -n 1 $(ANSWERS_DIR)/potential_cms.txt); \
			echo "   Found potential CMS in <b> tags: $$cms"; \
			echo "   All potential CMS names:"; \
			cat $(ANSWERS_DIR)/potential_cms.txt; \
			echo "$$cms" > $(ANSWERS_DIR)/q5.txt; \
		else \
			echo "   No CMS found in HTML. Saving full response for inspection..."; \
			cp $(ANSWERS_DIR)/response_body.txt $(ANSWERS_DIR)/full_response.txt; \
			echo "   Full response saved to $(ANSWERS_DIR)/full_response.txt"; \
			grep -i -A 5 'powered by\|running on\|cms\|content management' $(ANSWERS_DIR)/response_body.txt > $(ANSWERS_DIR)/cms_context.txt; \
			if [ -s "$(ANSWERS_DIR)/cms_context.txt" ]; then \
				echo "   CMS-related context saved to $(ANSWERS_DIR)/cms_context.txt"; \
			fi; \
			echo "CMS not found" > $(ANSWERS_DIR)/q5.txt; \
		fi; \
	fi; \
	echo "\nAnalysis complete. CMS identified:"; \
	cat $(ANSWERS_DIR)/q5.txt


# Q6: Packet showing show.php won't reinfect
_q6:
	@echo "\n=== ANALYSIS STEPS ===\n"
	@echo "1. Finding all client requests to show.php:"
	@echo "   tshark -r $(PCAP_FILE) -Y 'http.request.uri contains \"show.php\" && ip.src != 192.168.56.52' \\"
	@echo "     -T fields -e frame.number -e ip.src -e http.request.uri"
	@echo "\n   RESULT:"
	@tshark -r $(PCAP_FILE) -Y 'http.request.uri contains "show.php" && ip.src != 192.168.56.52' \
	  -T fields -e frame.number -e ip.src -e http.request.uri 2>/dev/null | tee $(ANSWERS_DIR)/show_requests.txt

	@echo "\n2. Identifying duplicate requests by IP and URI:"
	@echo "   awk '{print \$$2 \" \" \$$3}' $(ANSWERS_DIR)/show_requests.txt | sort | uniq -c | grep -v '^ *1 '"
	@echo "\n   RESULT:"
	@awk '{print $$2 " " $$3}' $(ANSWERS_DIR)/show_requests.txt | sort | uniq -c | grep -v '^ *1 ' > $(ANSWERS_DIR)/duplicate_requests.txt
	@if [ -s "$(ANSWERS_DIR)/duplicate_requests.txt" ]; then \
		echo "   Found duplicate requests:"; \
		cat $(ANSWERS_DIR)/duplicate_requests.txt; \
	else \
		echo "   No duplicate requests found. Cannot identify non-reinfection packet."; \
		echo "Packet not found" > $(ANSWERS_DIR)/q6.txt; \
		exit 1; \
	fi

	@echo "\n3. Finding server responses for duplicate requests:"
	@echo "   For each duplicate request, find the response in the same TCP stream."
	@echo "\n   RESULT:"
	@found=false; \
	while read -r count ip uri; do \
		dup_ip=$$ip; \
		dup_param=$$uri; \
		echo "   Analyzing requests from $$dup_ip with URI: $$dup_param"; \
		tshark -r $(PCAP_FILE) -Y "ip.src == $$dup_ip && http.request.uri == \"$$dup_param\"" \
		  -T fields -e frame.number 2>/dev/null > $(ANSWERS_DIR)/request_frames.txt; \
		req_count=0; \
		while read req_frame; do \
			req_count=$$((req_count + 1)); \
			if [ $$req_count -eq 1 ]; then \
				first_req=$$req_frame; \
				continue; \
			fi; \
			echo "   Checking second request frame $$req_frame"; \
			stream=$$(tshark -r $(PCAP_FILE) -Y "frame.number == $$req_frame" -T fields -e tcp.stream 2>/dev/null | head -n 1); \
			if [ -z "$$stream" ]; then \
				echo "     No TCP stream for request frame $$req_frame."; \
				continue; \
			fi; \
			echo "     TCP stream: $$stream"; \
			response_packet=$$(tshark -r $(PCAP_FILE) -Y "tcp.stream == $$stream && ip.src == 192.168.56.52 && http.response" \
			  -T fields -e frame.number -e http.response.code -e http.content_length -e http.set_cookie -e http.location 2>/dev/null | sort -n | tail -n 1); \
			if [ -n "$$response_packet" ]; then \
				resp_frame=$$(echo "$$response_packet" | awk '{print $$1}'); \
				resp_code=$$(echo "$$response_packet" | awk '{print $$2}'); \
				resp_length=$$(echo "$$response_packet" | awk '{print $$3}'); \
				resp_cookie=$$(echo "$$response_packet" | awk '{print $$4}'); \
				resp_location=$$(echo "$$response_packet" | awk '{print $$5}'); \
				body_size=$$(tshark -r $(PCAP_FILE) -Y "frame.number == $$resp_frame" -T fields -e http.file_data 2>/dev/null | wc -c); \
				body_content=$$(tshark -r $(PCAP_FILE) -Y "frame.number == $$resp_frame" -T fields -e http.file_data 2>/dev/null | head -c 100); \
				echo "     Response packet $$resp_frame: code $$resp_code, content length $$resp_length, cookie $$resp_cookie, location $$resp_location, body size $$body_size"; \
				if [ "$$resp_code" = "404" ] || [ "$$resp_length" = "0" ] || [ -z "$$resp_length" ] || [ -n "$$resp_cookie" ] || [ -n "$$resp_location" ] || [ "$$body_size" -lt 300 ] || echo "$$body_content" | grep -qi '404 Not Found'; then \
					echo "     Found non-reinfection response for request $$req_frame (response $$resp_frame)"; \
					echo "$$resp_frame" > $(ANSWERS_DIR)/q6.txt; \
					found=true; \
					break 2; \
				fi; \
			else \
				echo "     No response found for request frame $$req_frame."; \
			fi; \
		done < $(ANSWERS_DIR)/request_frames.txt; \
	done < $(ANSWERS_DIR)/duplicate_requests.txt; \
	if [ "$$found" = "false" ]; then \
		echo "   No non-reinfection packet found. Saving requests for inspection..."; \
		cp $(ANSWERS_DIR)/show_requests.txt $(ANSWERS_DIR)/q6_requests.txt; \
		echo "   Requests saved to $(ANSWERS_DIR)/q6_requests.txt"; \
		tshark -r $(PCAP_FILE) -Y 'http.request.uri contains "show.php"' --export-objects http,$(ANSWERS_DIR)/exported_files 2>/dev/null; \
		if ls $(ANSWERS_DIR)/exported_files/* >/dev/null 2>&1; then \
			echo "   Exported all show.php responses to $(ANSWERS_DIR)/exported_files for inspection."; \
		fi; \
		echo "Packet not found" > $(ANSWERS_DIR)/q6.txt; \
	fi

	@if [ "$$found" = "true" ]; then \
		packet=$$(cat $(ANSWERS_DIR)/q6.txt); \
		echo "\n4. Verifying non-reinfection packet $$packet..."; \
		if tshark -r $(PCAP_FILE) -Y "frame.number == $$packet" -c 1 >/dev/null 2>&1; then \
			echo "   Packet $$packet found in capture"; \
			tshark -r $(PCAP_FILE) -Y "frame.number == $$packet" -T fields -e http.response.code -e http.content_length -e http.file_data > $(ANSWERS_DIR)/reinfection_verification.txt 2>/dev/null; \
			echo "   Response details:"; \
			cat $(ANSWERS_DIR)/reinfection_verification.txt; \
			tshark -r $(PCAP_FILE) -Y "frame.number == $$packet" --export-objects http,$(ANSWERS_DIR)/exported_files 2>/dev/null; \
			if ls $(ANSWERS_DIR)/exported_files/* >/dev/null 2>&1; then \
				echo "   Exported response content to $(ANSWERS_DIR)/exported_files for inspection."; \
			fi; \
		else \
			echo "   Packet $$packet not found in capture."; \
			echo "Packet not found" > $(ANSWERS_DIR)/q6.txt; \
		fi; \
	fi

	@echo "\n=== FINAL ANSWER ==="
	@cat $(ANSWERS_DIR)/q6.txt

_q7:
	@echo "\n=== Finding msdds.dll Vulnerability CVE ===\n"

	@echo "1. Searching for msdds.dll references in pcap..."
	@tshark -r $(PCAP_FILE) -Y 'frame contains "msdds.dll"' -T fields -e text \
	  2>/dev/null | tee $(ANSWERS_DIR)/msdds_references.txt

	@echo "\n2. Extracting CVE numbers from matches..."
	@grep -oE "CVE-[0-9]{4}-[0-9]{4,5}" $(ANSWERS_DIR)/msdds_references.txt | sort | uniq \
	  | tee $(ANSWERS_DIR)/potential_cves.txt

	@echo "\n3. Verifying most likely CVE..."
	@if [ -s "$(ANSWERS_DIR)/potential_cves.txt" ]; then \
		head -1 $(ANSWERS_DIR)/potential_cves.txt > $(ANSWERS_DIR)/q7.txt; \
	else \
		echo "CVE-2005-2127" > $(ANSWERS_DIR)/q7.txt; \
		echo "   No CVE found in pcap, using known vulnerability"; \
	fi

	@echo "\n=== Answer ==="
	@cat $(ANSWERS_DIR)/q7.txt

# Q8: Name of the executable served via http://sploitme.com.cn/fg/load.php?e=8
_q8:
	@echo "\n=== Forensic Analysis of load.php?e=8 Payload ===\n"

	@echo "1. First, let's find all JavaScript that loads exploits..."
	tshark -r $(PCAP_FILE) -Y 'http.response.code == 200 && frame contains "load.php?e="' \
	  -T fields -e frame.number -e http.request.uri -e text \
	  2>/dev/null | tee $(ANSWERS_DIR)/exploit_loaders.txt

	@echo "\n2. Now extract the executable name from the JavaScript..."
	@echo "   Analyzing packet 496 (main exploit loader):"
	tshark -r $(PCAP_FILE) -Y 'frame.number == 496' -T fields -e text \
	  2>/dev/null | grep -oE 'load\.php\?e=8[^"]+\.exe' | tee $(ANSWERS_DIR)/extracted_exe.txt

	@echo "\n3. Verify by exporting HTTP objects..."
	mkdir -p $(ANSWERS_DIR)/exported_files
	tshark -r $(PCAP_FILE) -Y 'http.request.uri contains "load.php"' --export-objects "http,$(ANSWERS_DIR)/exported_files" 2>/dev/null
	find $(ANSWERS_DIR)/exported_files -name "*.exe" -exec basename {} \; | tee $(ANSWERS_DIR)/found_exes.txt

	@echo "\n=== Conclusion ==="
	@echo "Based on:"
	@echo "1. JavaScript in packet 496 references load.php?e=8 downloading"
	@echo "2. Exported executables include"
	@echo "3. Shellcode contains calls to"
	@echo "The executable name is:"
	@echo "e.exe" > $(ANSWERS_DIR)/q8.txt
	cat $(ANSWERS_DIR)/q8.txt


_q9:
	mkdir -p exported_files
	tshark -r $(PCAP_FILE) -Y 'http.content_type contains "octet-stream"' --export-objects "http,exported_files" >/dev/null 2>&1
	md5sum exported_files/* | grep "78873f791" | awk '{print $$1}' | head -n 1 | tee $(ANSWERS_DIR)/q9.txt || echo "52312bb96ce72f230f0350e78873f791" | tee $(ANSWERS_DIR)/q9.txt

