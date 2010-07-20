#!/usr/bin/ruby -Ku

require 'rubygems'
require 'open-uri'
require 'rss/2.0'
require 'mysql'

require 'optparse'
opt = {:host=>'localhost', :user=>'root', :passwd=>'', :dbname=>'googlenews', :port=>3306}
parser = OptionParser.new
#parser.banner = "Usage: #$0 [options] trainfile modelfile"
parser.on('--host=VAL', String, 'database host') {|v| opt[:host] = v }
parser.on('--user=VAL', String, 'database user') {|v| opt[:user] = v }
parser.on('--password=VAL', String, 'database password') {|v| opt[:passwd] = v }
parser.on('--dbname=VAL', String, 'database name') {|v| opt[:dbname] = v }
parser.on('--port=VAL', Integer, 'database port') {|v| opt[:port] = v }
parser.parse!(ARGV)


# Database
# create database googlenews character set utf8;
# create table news (id int auto_increment, title varchar(1024), lang varchar(8), body text, primary key (id));
# create index news_title on news (title);
# create index news_lang on news (lang);
db = Mysql::init
db.options Mysql::SET_CHARSET_NAME, 'utf8'
db.real_connect opt[:host], opt[:user], opt[:passwd], opt[:dbname], opt[:port]
ps_select = db.prepare("select id from news where title=?")
ps_insert = db.prepare("insert into news (title,lang,body) values (?,?,?)")


langlist = [
  # 日本語,中国語(簡体字),中国語(繁体字),韓国語,英語,フランス語,
  # イタリア語,スペイン語,ロシア語,アラビア語,ベトナム語,タイ語,
  "ja", "zh-CN", "zh-TW", "ko", "en", "fr", "it", "es", "ru", "ar", "vi", "th",
  # ドイツ語,ヒンディー語,ポルトガル語,インドネシア語,
  "de", "hi", "pt-PT", "id",
  # "bn" # ベンガル語
  # "fa" # ペルシャ語
]
def rssurl(lang)
  if lang=="ja"
    'http://news.google.com/news?hl=ja&ned=us&ie=UTF-8&oe=UTF-8&output=rss'
  else
    "http://news.google.com/news?pz=1&cf=all&hl=#{lang}&output=rss"
  end
end

langlist.each do |lang|
  url = rssurl(lang)
  #puts url
  rss = open(url) {|f| RSS::Parser.parse(f.read, false) }

  rss.items.each do |item|
    rs = ps_select.execute(item.title)
    if !rs.fetch
      body = item.description.gsub(/<[^>]*>/, ' ').gsub(/&nbsp;/, ' ').gsub(/[ \t]+/, ' ')
      ps_insert.execute item.title, lang, body
    end
  end
  sleep 1
end

