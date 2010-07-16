#!/usr/bin/ruby -Ku

require 'rubygems'
require 'open-uri'
require 'rss/2.0'
require 'mysql'

require 'optparse'
opt = {
  :host=>'localhost', :user=>'root', :passwd=>'', :dbname=>'googlenews', :port=>3306,
  :model=>'model'
}
parser = OptionParser.new
#parser.banner = "Usage: #$0 [options] trainfile modelfile"
parser.on('--host=VAL', String, 'database host') {|v| opt[:host] = v }
parser.on('--user=VAL', String, 'database user') {|v| opt[:user] = v }
parser.on('--password=VAL', String, 'database password') {|v| opt[:passwd] = v }
parser.on('--dbname=VAL', String, 'database name') {|v| opt[:dbname] = v }
parser.on('--port=VAL', Integer, 'database port') {|v| opt[:port] = v }
parser.on('-f VAL', String, 'model filename') {|v| opt[:model] = v }
parser.parse!(ARGV)

LANGLIST = [
  # 日本語,中国語(繁体字),中国語(簡体字),韓国語,英語,フランス語,
  # イタリア語,スペイン語,ロシア語,アラビア語,ベトナム語,タイ語,
  "ja", "zh-CN", "zh-TW", "ko", "en", "fr", "it", "es", "ru", "ar", "vi", "th",
  # ドイツ語,ヒンディー語,ポルトガル語,インドネシア語,
  "de", "hi", "pt-PT", "id", 
  # "bn" # ベンガル語
  # "fa" # ペルシャ語
]

# model
class Detector
  def initialize(filename, alpha=0.1, beta=1.0)
    @n_k, @p_ik = open(filename){|f| Marshal.load(f) }
    @p_ik.default = 0
    @alpha = alpha
    @beta = beta
  end
  def init
    @prob = Hash.new
    LANGLIST.each {|lang| @prob[lang] = 1.0 }
    @maxprob = 0
  end
  def append(x)
    return unless @p_ik.key?(x)
    freq = @p_ik[x]
    sum = 0
    LANGLIST.each do |lang|
      @prob[lang] *= (freq[lang] + @alpha) / (@n_k[lang] + @beta)
      sum += @prob[lang]
    end
    @maxprob = 0
    LANGLIST.each do |lang|
      @prob[lang] /= sum
      @maxprob = @prob[lang] if @maxprob < @prob[lang]
    end
  end
  def maxprob; @maxprob; end
  def problist; @prob.to_a.select{|x| x[1]>0.1}.sort_by{|x| -x[1]}; end
end
detector = Detector.new(opt[:model])


# Database
db = Mysql::init
db.options Mysql::SET_CHARSET_NAME, 'utf8'
db.real_connect opt[:host], opt[:user], opt[:passwd], opt[:dbname], opt[:port]
ps_select = db.prepare("select id,title,lang,body from news order by lang")

ps_select.execute
count = Hash.new(0)
correct = Hash.new(0)
while rs = ps_select.fetch
  id, title, lang, body = rs
  text = title + "\n" + body
  xlist = text.split(//)

  pre_x = " "
  detector.init
  xlist.each do |x|
    if x[0] <= 32
      x = " "
    elsif x =~ /^[\xd0-\xd1][\x80-\xbf]/      # Cyrillic
      x = "\xd0\x96"
    elsif x =~ /^[\xd8-\xd9][\x80-\xbf]/      # Arabic
      x = "\xd8\xa6"
    elsif x =~ /^\xe0[\xa4-\xa5][\x80-\xbf]/  # Devanagari
      x = "\xe0\xa4\x85"
    elsif x =~ /^\xe0[\xb8-\xb9][\x80-\xbf]/  # Thai
      x = "\xe0\xb9\x91"
    elsif x =~ /^\xe1[\xba-\xbb][\x80-\xbf]/  # Latin Extended Additional(Vietnam)
      x = "\xe1\xba\xa1"
    elsif x =~ /^\xe3[\x81-\x83][\x80-\xbf]/  # Hiragana / Katakana
      x = "\xe3\x81\x82"
    elsif x =~ /^\xea[\xb0-\xbf][\x80-\xbf]/  # Hangul Syllables 1
      x = "\xea\xb0\x80"
    elsif x =~ /^[\xeb-\xed][\x80-\xbf]{2}/   # Hangul Syllables 2
      x = "\xed\x9e\x98"
    end

    x2 = pre_x + x
    detector.append x if x != " "
    detector.append x2 if x2 != "  "
    break if detector.maxprob > 0.99
  end

  problist = detector.problist
  puts "#{id},#{lang},#{title},#{problist.inspect}"
  count[lang] += 1
  correct[lang] += 1 if problist[0][0] == lang
end

count.keys.sort.each do |lang|
  puts "#{lang} #{correct[lang]} / #{count[lang]} (#{(10000.0 * correct[lang] / count[lang]).to_i / 100.0})"
end

