/*
 * Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */
package com.datastax.dse.demos.solr.commands;

import java.util.List;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.SolrServer;
import org.apache.solr.client.solrj.impl.HttpSolrServer;
import org.apache.solr.common.SolrDocumentList;
import com.google.common.base.Joiner;

public class LucenePerfCmdRunner extends CmdRunner
{
    public final static String CMD_STRING_ID = "LUCPERF";
    private final SolrServer readSolrClient;
    
    LucenePerfCmdRunner(String fullUrl)
    {
        this.readSolrClient = new HttpSolrServer(fullUrl);
    }
    
    public Type runCommand(List<String> cmd2Run) throws Throwable
    {
        SolrQuery query = new SolrQuery();

        String lucCmd = cmd2Run.get(1);
        // Remove comments
        String lucCmdText = lucCmd.replaceAll("#.*", "").trim();
        lucCmdText = lucCmdText.substring(lucCmdText.indexOf(':') + 1, lucCmdText.length()).trim();
        String category = lucCmd.substring(0, lucCmd.indexOf(':')).trim();

        switch (category)
        {
            case "HighTerm":
            case "MedTerm":
            case "LowTerm":
            case "Prefix3":
            case "Wildcard":
            case "Fuzzy1":
            case "Fuzzy2":
            case "HighPhrase":
            case "HighSloppyPhrase":
            case "MedPhrase":
            case "MedSloppyPhrase":
            case "LowPhrase":
            case "LowSloppyPhrase":
                query.set("q", "body:" + lucCmdText);
                break;

            case "AndHighHigh":
            case "AndHighMed":
            case "AndHighLow":
                String[] andParts = lucCmdText.split("\\s");
                for (int i = 0; i < andParts.length; i++)
                {
                    // We build the query part and also remove the preceding '+' char
                    andParts[i] = "body:" + andParts[i].substring(1);
                }

                String joinedAndQuery = Joiner.on(" AND ").join(andParts);
                query.set("q", joinedAndQuery);
                break;

            case "OrHighHigh":
            case "OrHighMed":
            case "OrHighLow":
                String[] orParts = lucCmdText.split("\\s");
                for (int i = 0; i < orParts.length; i++)
                {
                    orParts[i] = "body:" + orParts[i];
                }

                String joinedOrQuery = Joiner.on(" OR ").join(orParts);
                query.set("q", joinedOrQuery);
                break;

            case "IntNRQ":
                String[] numericRangeParts = lucCmdText.split("\\s");
                String fieldName = numericRangeParts[0].replaceAll("nrq//", "");
                int start = Integer.parseInt(numericRangeParts[1]);
                int end = Integer.parseInt(numericRangeParts[2]);

                query.set("q", fieldName + ":[" + start + " TO " + end + "]");
                break;
                
            case "HighSpanNear":
            case "MedSpanNear":
            case "LowSpanNear":
            case "Respell":
                System.err.println("WARN: Solr doesn't support this category: " + category);
            default:
                throw new Exception("Unknown lucene task: " + category + " please remove and try again.");
        }

        SolrDocumentList results = readSolrClient.query(query).getResults();
        numDocsRead = results.getNumFound();

        return Type.READ;
    }

    @Override
    public void close() throws Throwable
    {
        readSolrClient.shutdown();
    }
    
}
