#!/usr/bin/ruby -Ku

require 'rubygems'
require 'open-uri'
require 'rss/2.0'


langlist = [
  # 中国語(繁体字),中国語(簡体字),韓国語,英語,フランス語,イタリア語,
  # スペイン語,ロシア語,アラビア語,ベトナム語,タイ語,
  "zh-CN", "zh-TW", "ko", "en", "fr", "it", "es", "ru", "ar", "vi", "th",
  # ドイツ語,ヒンディー語,ベンガル語,ポルトガル語,インドネシア語,
  "de", "hi", "bn", "pt-PT", "id", 
  # 日本語,ペルシャ語
  #"ja", "fa"
]

langlist.each do |lang|
  rss = open("http://news.google.com/news?pz=1&cf=all&hl=#{lang}&output=rss") do |f|
    RSS::Parser.parse(f.read, false)
  end

  rss.items.each do |item|
    puts item.title
    puts item.description.gsub(/<[^>]*>/, ' ').gsub(/&nbsp;/, ' ').gsub(/\s\s+/, ' ')
  end
end

