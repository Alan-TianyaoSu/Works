package com.truiton.autosuggesttextviewapicall

import android.content.Context
import android.widget.ArrayAdapter
import android.widget.Filter
import android.widget.Filterable

class AutoSuggestAdapter(context: Context, resource: Int) :
    ArrayAdapter<String>(context, resource), Filterable {

    private var mListData: MutableList<String> = mutableListOf()

    fun setData(list: List<String>) {
        mListData.clear()
        mListData.addAll(list)
    }

    override fun getCount(): Int {
        return mListData.size
    }

    override fun getItem(position: Int): String? {
        return mListData[position]
    }

    override fun getFilter(): Filter {
        return object : Filter() {
            override fun performFiltering(constraint: CharSequence?): FilterResults {
                val results = FilterResults()
                results.values = mListData
                results.count = mListData.size
                return results
            }

            override fun publishResults(constraint: CharSequence?, results: FilterResults?) {
                notifyDataSetChanged()
            }
        }
    }
}