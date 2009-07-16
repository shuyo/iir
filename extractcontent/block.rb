#!/usr/bin/ruby -Ku

require 'rubygems'
require 'hpricot'
require "extractcontent.rb"

TAGLIST = [
"div","h1","h2","h3","h4","h5","h6",
"a","br","img","li","tr","span","b","font","p","table","input","option","dd","td","script","dt","ul","em","strong","th","small","form","dl","hr","wbr","cite","body","label","noscript","area","tbody","i","iframe","nobr","button","select","style","param","blockquote","scr","textarea","rdf","u","ol","o","center","col","address","tt","link","map","sup","embed","object","v","fieldset","rp","colgroup","sc","n","meta","legend","optgroup","caption","abbr","code","thead","big","pre","rb","ruby","rt","del","title","html","comment","s","fn","head","frame","sub","strike","marquee","spacer","keeper","smoothie","d","name","id","frameset","acronym","author","updated","file","noframes","this","dfn","fb","ins","csobj","len","basefont","noindex","category","tfoot","content","obj","entry","ifr","scri","limit","var","t","bold","linklist","feed","kbd","http","l","im","nolayer","bu","document","w","com","blink","ilayer","c","msearch","image","place","csaction","base","arimgname","darea","q"]

CHARLIST = "!\"\#$%&'()-=~^|\\[]{}<>@`+*;:,.?_/、。・！？「」『』【】"

SEPARATORS = {
  /<!--\s*google_ad_section_start/ => :body,
  /<!--\s*google_ad_section_end/ => :unclassified,
  /<!--\s*extractcontent_title/ => :body,
  /<!--\s*extractcontent_body/ => :body,
  /<!--\s*extractcontent_bbs/ => :body,   # 本文性の高いコメント
  /<!--\s*extractcontent_abstract/ => :abstract,  # ページやサイトの概要
  /<!--\s*extractcontent_header/ => :menu,  #:header,
  /<!--\s*extractcontent_footer/ => :menu,  #:footer,
  /<!--\s*extractcontent_comment/ => :comment,   # コメント＆トラックバック
  /<!--\s*extractcontent_form/ => :menu,         # 入力フォーム
  /<!--\s*extractcontent_search/ => :menu,         # 検索フォーム
  /<!--\s*extractcontent_menu/ => :menu,
  /<!--\s*extractcontent_linklist/ => :linklist,  #内容に必ずしも関係ない記事へのリンク
  /<!--\s*extractcontent_cm/ => :cm,  #広告
  /<!--\s*extractcontent_end/ => :unclassified,
}

ID_KEYWORDS = {
  /menu|footer|header|bottom|copyright|navi/i => :menu,
  /recommend/i => :cm,
  /comment|trackback/i => :comment,
}

Extractor = ExtractContent::Extractor.new

class ExtractFeature
  def initialize
    @all_tags = Hash.new(0)
  end
  attr_reader :all_tags

  def analyze(filename)
    puts "# " + filename
    html = open(filename){|f| f.read}

    #puts Extractor.analyse(html)

    if html =~ /<\/head\s*>/im
      html = $' #'
    end

    list = html.split(/(<(?:div|center|t[dr]|h[1-6]|[udo]l|p)[^>]*>|<br\s*\/?>\s*<br\s*\/?>)/i)
    #puts list.join("\n====================================================================\n")

    current_class = :unclassified
    result = []
    list.insert 0, ""
    while list.length>0
      block = list.shift + list.shift

      # tag count
      block_tags = Hash.new(0)
      block.scan(/<([A-Za-z][A-Za-z0-9]*)/) do |tag|
        tag = tag[0].downcase
        block_tags[tag] += 1
        @all_tags[tag] += 1
      end
      features = Hash.new(0)
      TAGLIST.each_with_index{|tag, i| features[i] = block_tags[tag] if block_tags.key?(tag) }

      # detect class
      post_class = now_class = nil
      SEPARATORS.each do |regsep, clz|
        if regsep =~block
          if $`.length > $'.length
            $stderr.puts "specified already post_class at #{filename}" if post_class
            post_class = clz
          else
            $stderr.puts "specified already now_class at #{filename}" if now_class
            now_class = clz
          end
        end
      end

      current_class = now_class if now_class
      if now_class || result.length == 0 || features.size > 1
        result << {:class=>current_class, :features=>features, :body=>block}

        #puts "============================================================================="
        #puts "#{current_class} #{features.map{|key,value| "#{key}:#{value}"}.join(" ")}"
        #puts block
      else
        pre_features = result[-1][:features]
        features.each{|key, value| pre_features[key] += value }
        result[-1][:body] = block
        #puts "++++++++"
        #puts block
      end

      current_class = post_class if post_class
    end
    result
  end
end

def classify(data)
  data.each do |item|
    if item[:class]==:unclassified
      item[:body].scan(/(?:id|class)=['"]([a-z0-9_ ]+)['"]/i) do |x|
        #
      end
    end
  end
  #
end

extractor = ExtractFeature.new
ARGV.each do |dir|
  Dir.foreach(dir) do |filename|
    next if filename =~ /^\./
    data = extractor.analyze dir + '/' + filename
    classify data
    data.each do |item|
      puts "#{item[:class]} #{item[:features].map{|key,value| "#{key}:#{value}"}.join(' ')}"
    end
  end
end

#puts "all tags = \n#{extractor.all_tags.to_a.sort{|a,b|b[1]<=>a[1]}.map{|a|"#{a[0]},#{a[1]}"}.join("\n")}"


=begin
### TODO
- external link
- nofollow link
- link in page #
- <a name
- using div id
- img alt
- date
=end
