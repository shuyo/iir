#!/usr/bin/env ruby
# -*- coding: utf-8 -*-

#require 'rubygems'
require 'open-uri'
require 'rss/2.0'

require 'common.rb'
require 'detect.rb'

LD::optparser.parse!(ARGV)
db = LD::db_connect

# Database
# create database googlenews character set utf8;
# create table news (id int auto_increment, title varchar(1024), lang varchar(8), body text, primary key (id));
# create index news_title on news (title);
# create index news_lang on news (lang);
ps_select = db.prepare("select id from news where title=?")
ps_insert = db.prepare("insert into news (title,lang,body) values (?,?,?)")

# Google News RSS
def rssurl(lang)
  if lang=="ja"
    'http://news.google.com/news?hl=ja&ned=us&ie=UTF-8&oe=UTF-8&output=rss'
  else
    "http://news.google.com/news?pz=1&cf=all&hl=#{lang}&output=rss"
  end
end

langlist = LanguageDetector::LANGLIST

langlist.each do |lang|
  url = rssurl(lang)
  #puts url
  rss = open(url) {|f| RSS::Parser.parse(f.read, false) }

  rss.items.each do |item|
    rs = ps_select.execute(item.title)
    if !rs.fetch
      body = item.description.gsub(/<nobr>.*?<\/nobr>/, '').gsub(/<[^>]*>/, ' ').gsub(/&nbsp;/, ' ').gsub(/[ \t]+/, ' ')
      ps_insert.execute item.title, lang, body
    end
  end
  sleep 1
end

