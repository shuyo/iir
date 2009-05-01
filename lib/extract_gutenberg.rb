#!/usr/bin/ruby

module Gutenberg
def self.extract(text)
  text = text.gsub(/[ \r]+$/, "") + "\n\n"
  $stderr.puts "Warning: HTML-formed comment in #{path}"if text.gsub!(/<-- .+? -->/m, "")
  $stderr.puts "Warning: HTML tag in #{path}"if text.gsub!(/<HTML>.+?<\/HTML>/mi, "")

  negative_phrase = /http|internet|project gutenberg|mail|ocr/i
  separator = /^(?:.+?END\*{1,2}|\*{3} START OF THE PROJECT GUTENBERG E(?:BOOK|TEXT).*? \*{3}|\*{9}END OF .+?|\*{3} END OF THE PROJECT GUTENBERG E(?:BOOK|TEXT).+?|\*{3}START\*.+\*START\*{3}|\**\s*This file should be named .+|\*{5}These [eE](?:Books|texts) (?:Are|Were) Prepared By .+\*{5})$/

	while text =~ separator
	  pre, post = $`, $'
	  text = if pre.length > post.length*3 then
	    pre
	  elsif post.length > pre.length*3 then
	    post
	  elsif pre.scan(negative_phrase).length < post.scan(negative_phrase).length
	    pre
	  else
	    post
	  end
	end

  text.gsub!(/^(?:Executive Director's Notes:|\[?Transcriber's Note|PREPARER'S NOTE|\[Redactor's note|\{This e-text has been prepared|As you may be aware, Project Gutenberg has been involved with|[\[\*]Portions of this header are|A note from the digitizer|ETEXT EDITOR'S BOOKMARKS|\[NOTE:|\[Project Gutenberg is|INFORMATION ABOUT THIS E-TEXT EDITION\n+|If you find any errors|This electronic edition was|Notes about this etext:|A request to all readers:|Comments on the preparation of the E-Text:|The base text for this edition has been provided by).+?\n(?:[\-\*]+)?\n\n/mi, "")
  text.gsub!(/^[\[\n](?:[^\[\]\n]+\n)*[^\n]*(?:Project\sGutenberg|\setext\s|\s[A-Za-z0-9]+@[a-z\-]+\.(?:com|net))[^\n]*(?:\n[^\[\]\n]+)*[\]\n]$/i, "")
  text.gsub!(/\{The end of etext of .+?\}/, "")
  text = text.strip + "\n\n"

	text.gsub!(/^(?:(?:End )?(?:of ?)?(?:by |This |The )?Project Gutenberg(?:'s )?(?:Etext)?|This (?:Gutenberg )?Etext).+?\n\n/mi, "")
	text.gsub!(/^(?:\(?E?-?(?:text )?(?:prepared|Processed|scanned|Typed|Produced|Edited|Entered|Transcribed|Converted|created) by|Transcribed from|Scanning and first proofing by|Scanned and proofed by|This e-text|This EBook of|Scanned with|This Etext created by|This eBook was (?:produced|updated) by|Image files scanned in by|\[[^\n]*mostly scanned by|This text was prepared by).+?\n\n/mi, "")

  if text=~/gutenberg|\setext\s|scanner|David Reed/i
	  $stderr.puts "Warning:  remain '#{$&.strip}'"
  elsif text=~/[^\s\*]@[^\s]+\./
	  $stderr.puts "Warning:  maybe remain mail adress"
  elsif text.length < 1024
	  $stderr.puts "Warning:  too small body"
	end

  text
end
end

puts Gutenberg.extract(ARGF.read) if $0 == __FILE__
