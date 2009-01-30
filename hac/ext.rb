#!/usr/bin/ruby
# ruby ext.rb ohenry/1444.zip ohenry/1646.zip ohenry/1725.zip ohenry/2777.zip ohenry/2776.zip

DEST_DIR = "output"
DEBUG = false

require '../lib/infinitive.rb'
INF = Infinitive.new
def extract_body(text)
  lang = "English"
  if text =~ /^Language:\s*([A-Za-z]+)\s*$/ && $1 != lang
    lang = $1
    $stderr.puts "Warning: #{path} in #{lang}"
  end

  text = text.gsub(/[ \r]+$/, "") + "\n\n"
  $stderr.puts "Warning: HTML-formed comment in #{path}"if text.gsub!(/<-- .+? -->/m, "")
  $stderr.puts "Warning: HTML tag in #{path}"if text.gsub!(/<HTML>.+?<\/HTML>/mi, "")
  r = /http|internet|project gutenberg|mail|ocr/i
	while text =~ /^(?:.+?END\*{1,2}|\*{3} START OF THE PROJECT GUTENBERG E(?:BOOK|TEXT).*? \*{3}|\*{9}END OF .+?|\*{3} END OF THE PROJECT GUTENBERG E(?:BOOK|TEXT).+?|\*{3}START\*.+\*START\*{3}|\**\s*This file should be named .+|\*{5}These [eE](?:Books|texts) (?:Are|Were) Prepared By .+\*{5})$/
	  puts $& if DEBUG
	  pre, post = $`, $'
	  text = if pre.length > post.length*3 then
	    pre
	  elsif post.length > pre.length*3 then
	    post
	  elsif pre.scan(r).length < post.scan(r).length
	    pre
	  else
	    post
	  end
	end

	puts "---- #{text.length}", text[0..100], text[-100..-1] if DEBUG

  text.gsub!(/^(?:Executive Director's Notes:|\[?Transcriber's Note|PREPARER'S NOTE|\[Redactor's note|\{This e-text has been prepared|As you may be aware, Project Gutenberg has been involved with|[\[\*]Portions of this header are|A note from the digitizer|ETEXT EDITOR'S BOOKMARKS|\[NOTE:|\[Project Gutenberg is|INFORMATION ABOUT THIS E-TEXT EDITION\n+|If you find any errors|This electronic edition was|Notes about this etext:|A request to all readers:|Comments on the preparation of the E-Text:|The base text for this edition has been provided by).+?\n(?:[\-\*]+)?\n\n/mi, "")
  text.gsub!(/^[\[\n](?:[^\[\]\n]+\n)*[^\n]*(?:Project\sGutenberg|\setext\s|\s[A-Za-z0-9]+@[a-z\-]+\.(?:com|net))[^\n]*(?:\n[^\[\]\n]+)*[\]\n]$/i, "")
  text.gsub!(/\{The end of etext of .+?\}/, "")
  text = text.strip + "\n\n"

	puts "---- #{text.length}", text[0..100], text[-100..-1] if DEBUG

	text.gsub!(/^(?:(?:End )?(?:of ?)?(?:by |This |The )?Project Gutenberg(?:'s )?(?:Etext)?|This (?:Gutenberg )?Etext).+?\n\n/mi, "")
	text.gsub!(/^(?:\(?E?-?(?:text )?(?:prepared|Processed|scanned|Typed|Produced|Edited|Entered|Transcribed|Converted|created) by|Transcribed from|Scanning and first proofing by|Scanned and proofed by|This e-text|This EBook of|Scanned with|This Etext created by|This eBook was (?:produced|updated) by|Image files scanned in by|\[[^\n]*mostly scanned by|This text was prepared by).+?\n\n/mi, "")

  $stderr.puts "Warning: removed David Reed's block" if text.gsub!(/(?:Scanner's Notes).+?David Reed/m, "")

	puts "---- #{text.length}", text[0..100], text[-100..-1] if DEBUG

  if text=~/gutenberg|\setext\s|scanner|David Reed/i
	  puts "  ... remaining '#{$&.strip}' in #{path} !"
  elsif text=~/[^\s\*]@[^\s]+\./
	  puts "  ... maybe remain mail adress in #{path} !"
  elsif text.length < 1024
	  puts "  ... #{path} has been too small!"
	end
  text
end





def inf(x)
  #x.downcase
  INF.infinitive(x)
end

docs = Array.new
terms = Hash.new

doc_id = 0
ARGV.each do |path|
  file = path.dup
	file = $1 if path =~ /\/([^\/]+)$/
	text = nil
	if path =~ /\.zip$/i
	  file.sub!(/\.zip$/i, ".txt")
	  text = `unzip -cq #{path} "*.txt"`
    open("#{DEST_DIR}/#{file}.org", "w"){|f| f.write text}
	else
	  text = open(path){|f| f.read}
	end
  text = extract_body(text)
  open("#{DEST_DIR}/#{file}", "w"){|f| f.write text}

  list = text.split(/^[IVX]+\s*\.?$/)[1..-1]
  list = text.split(/^\n{4}$/) if list.size<=1
  list.each do |x|
    next unless x =~ /^(.+)$/
    title = $1

    words = x.scan(/[A-Za-z]+(?:'t)?/)
    next if words.size < 1000

    n = 0
    words.each do |word|
      word = inf(word)
      terms[word] ||= Hash.new(0)
      terms[word][doc_id] += 1
      n += 1
    end

    docs[doc_id] = {:title=>title, :n_words=>n}
    doc_id += 1
  end
end

if true
  require 'pstore'
  db = PStore.new('corpus')
  db.transaction do
    db[:docs] = docs
    db[:terms] = terms
  end

else
  p docs
  puts terms.size

  distrib = Array.new(docs.length, 0)
  terms.each do |word, x|
    distrib[x.size-1] += 1
  end
  distrib.each_with_index do |x, i|
    puts "#{i+1}, #{x}"
  end
end
