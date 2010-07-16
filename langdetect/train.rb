#!/usr/bin/ruby -Ku

require 'rubygems'
require 'open-uri'
require 'rss/2.0'
require 'mysql'

require 'optparse'
opt = {
  :host=>'localhost', :user=>'root', :passwd=>'', :dbname=>'googlenews', :port=>3306,
  :model=>'model', :N=>150,
}
parser = OptionParser.new
#parser.banner = "Usage: #$0 [options] trainfile modelfile"
parser.on('--host=VAL', String, 'database host') {|v| opt[:host] = v }
parser.on('--user=VAL', String, 'database user') {|v| opt[:user] = v }
parser.on('--password=VAL', String, 'database password') {|v| opt[:passwd] = v }
parser.on('--dbname=VAL', String, 'database name') {|v| opt[:dbname] = v }
parser.on('--port=VAL', Integer, 'database port') {|v| opt[:port] = v }
parser.on('-f VAL', String, 'model filename') {|v| opt[:model] = v }
parser.on('-n VAL', Integer, 'max size of training data') {|v| opt[:N] = v }
parser.parse!(ARGV)


# Database
db = Mysql::init
db.options Mysql::SET_CHARSET_NAME, 'utf8'
db.real_connect opt[:host], opt[:user], opt[:passwd], opt[:dbname], opt[:port]
ps_select = db.prepare("select title,lang,body from news order by rand()")

ps_select.execute
n_k = Hash.new(0)
p_ik = Hash.new{|h,k| h[k]=Hash.new(0)}
while rs = ps_select.fetch
  title, lang, body = rs
  next if n_k[lang] >= opt[:N]
  n_k[lang] += 1
  text = title + "\n" + body
  xlist = text.split(//)
  unigrams = Hash.new
  bigrams = Hash.new
  pre_x = " "
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
    unigrams[x] = 1 if x != " "
    bigrams[x2] = 1 if x2 != "  "
    pre_x = x
  end
  unigrams.each do |unigram, dummy|
    p_ik[unigram][lang] += 1
  end
  bigrams.each do |bigram, dummy|
    p_ik[bigram][lang] += 1
  end
end

=begin
langlist = [
  # 日本語,中国語(繁体字),中国語(簡体字),韓国語,英語,フランス語,
  # イタリア語,スペイン語,ロシア語,アラビア語,ベトナム語,タイ語,
  "ja", "zh-CN", "zh-TW", "ko", "en", "fr", "it", "es", "ru", "ar", "vi", "th",
  # ドイツ語,ヒンディー語,ポルトガル語,インドネシア語,
  "de", "hi", "pt-PT", "id", 
  # "bn" # ベンガル語
  # "fa" # ペルシャ語
]

puts ","+langlist.join(',')
p_ik.to_a.sort.each do |unigram,langs|
  langs.default = ''
  puts "'#{unigram.unpack('H*')[0]},#{langlist.map{|lang| langs[lang]}.join(',')}"
end
=end

p_ik.default = 0
open(opt[:model], 'w'){|f| Marshal.dump([n_k, p_ik], f) }

