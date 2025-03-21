### |||||||||||||||||||||||||||||||||||||| IMPORT PACKAGE  ||||||||||||||||||||||||||||||||||||||||||
import streamlit as st
import numpy as np
import pandas as pd
from pandas import DataFrame

### |||||||||||||||||||||||||||||||||||||| MAIN-INTERFACE ||||||||||||||||||||||||||||||||||||||||||

st.set_page_config(page_title="AMAS - Analytics software",
                   page_icon=":microscope:",
                   layout="wide"
                   )

st.sidebar.title(" :computer: :red[_AMAS_] ")
st.sidebar.write("version 1.2")

with st.container():
    st.subheader("Welcome to :red[AMAS] :tada:")
    st.title(":red[A] :red[M]aster's Degree Research :red[A]nalytics in :red[S]treamlit Program")
    st.write("[ Made by _Mr. Ran Kitkangplu_ at _MAE FAH LUANG UNIVERSITY_ ]")
    st.sidebar.write("## :blue[Pick your a raw data file] ##")
    uploaded_file = st.sidebar.file_uploader("Choice your data in this box",
                                             type=".csv"
                                             )

##|||||||||||||||||||||||||||||||||||||| UP_LOAD SAMPLE CSV_FILE ||||||||||||||||||||||||||||||||||||||||||

if uploaded_file:

    st.markdown("#### All raw data preview ####")
    df: DataFrame | None = pd.read_csv(uploaded_file)
    st.dataframe(df.head(100),
                 hide_index=True
                 )
    variables = df.columns
    variables = np.array(variables)
    st.sidebar.write("#### :green[Choice your SAMPLE column] ####")
    sam_select = st.sidebar.selectbox("Choice your sample form data this box ", variables)
    sam_data = df[sam_select]
    com_select = st.sidebar.selectbox("Choice your compound column form data this box ", variables)
    com_data = df[com_select].unique()

    st.markdown("#### Prepare your sample ####")
    sam_count = df.value_counts(sam_select)

    ### |||||||||||||||||||||||||||||||||||||| CHANGE SAMPLE NAME ||||||||||||||||||||||||||||||||||||||||||

    sel_sample = st.checkbox('I want to define sample name')
    if sel_sample:
        st.markdown("## Define your sample ##")
        tab_a, tab_b, tab_c, tab_d = st.columns([0.2, 0.2, 0.2, 0.2])
        tab_e, tab_f, tab_g, tab_h = st.columns([0.2, 0.2, 0.2, 0.2])
        tab_i, tab_j = st.columns([0.2, 0.2])
        st.markdown('---')

        with tab_a:
            dis_a_true = True
            dis_a = st.checkbox('Have A sample')
            if dis_a:
                dis_a_true = False
                a_sample_selected = st.selectbox("Which sample to define A sample name",
                                                 set(sam_data),
                                                 disabled=dis_a_true
                                                 )
                a_sample_name = st.text_input("Please name A sample",
                                              a_sample_selected
                                              )
                st.write("Define A sample is: ",
                         a_sample_selected
                         )
                st.write(":red[To be:]",
                         a_sample_name
                         )
                df = df.replace({a_sample_selected: a_sample_name})

        with tab_b:
            dis_b_true = True
            dis_b = st.checkbox('Have B sample')
            if dis_b:
                dis_b_true = False
                b_sample_selected = st.selectbox("Which sample to define B sample name",
                                                 set(sam_data),
                                                 disabled=dis_b_true
                                                 )
                b_sample_name = st.text_input("Please name B sample",
                                              b_sample_selected
                                              )
                st.write("Define B sample is: ",
                         b_sample_selected
                         )
                st.write(":red[To be:]",
                         b_sample_name
                         )
                df = df.replace({b_sample_selected: b_sample_name})

        with tab_c:
            dis_c_true = True
            dis_c = st.checkbox('Have C sample')
            if dis_c:
                dis_c_true = False
                c_sample_selected = st.selectbox("Which sample to define C sample name",
                                                 set(sam_data),
                                                 disabled=dis_c_true
                                                 )
                c_sample_name = st.text_input("Please name C sample",
                                              c_sample_selected
                                              )
                st.write("Define C sample is: ",
                         c_sample_selected
                         )
                st.write(":red[To be:]",
                         c_sample_name
                         )
                df = df.replace({c_sample_selected: c_sample_name})

        with tab_d:
            dis_d_true = True
            dis_d = st.checkbox('Have D sample')
            if dis_d:
                dis_d_true = False
                d_sample_selected = st.selectbox("Which sample to define D sample name",
                                                 set(sam_data),
                                                 disabled=dis_d_true
                                                 )
                d_sample_name = st.text_input("Please name D sample",
                                              d_sample_selected
                                              )
                st.write("Define D sample is: ",
                         d_sample_selected
                         )
                st.write(":red[To be:]",
                         d_sample_name
                         )
                df = df.replace({d_sample_selected: d_sample_name})

        with tab_e:
            dis_e_true = True
            dis_e = st.checkbox('Have E sample')
            if dis_e:
                dis_e_true = False
                e_sample_selected = st.selectbox("Which sample to define E sample name",
                                                 set(sam_data),
                                                 disabled=dis_e_true)
                e_sample_name = st.text_input("Please name E sample",
                                              e_sample_selected
                                              )
                st.write("Define E sample is: ",
                         e_sample_selected)
                st.write(":red[To be:]",
                         e_sample_name
                         )
                df = df.replace({e_sample_selected: e_sample_name})

        with tab_f:
            dis_f_true = True
            dis_f = st.checkbox('Have F sample')
            if dis_f:
                dis_f_true = False
                f_sample_selected = st.selectbox("Which sample to define F sample name",
                                                 set(sam_data),
                                                 disabled=dis_f_true
                                                 )
                f_sample_name = st.text_input("Please name F sample",
                                              f_sample_selected
                                              )
                st.write("Define F sample is: ",
                         f_sample_selected
                         )
                st.write(":red[To be:]",
                         f_sample_name
                         )
                df = df.replace({f_sample_selected: f_sample_name})

        with tab_g:
            dis_g_true = True
            dis_g = st.checkbox('Have G sample')
            if dis_g:
                dis_g_true = False
                g_sample_selected = st.selectbox("Which sample to define G sample name",
                                                 set(sam_data),
                                                 disabled=dis_g_true
                                                 )
                g_sample_name = st.text_input("Please name G sample",
                                              g_sample_selected
                                              )
                st.write("Define G sample is: ",
                         g_sample_selected
                         )
                st.write(":red[To be:]",
                         g_sample_name
                         )
                df = df.replace({g_sample_selected: g_sample_name})

        with tab_h:
            dis_h_true = True
            dis_h = st.checkbox('Have H sample')
            if dis_h:
                dis_h_true = False
                h_sample_selected = st.selectbox("Which sample to define H sample name",
                                                 set(sam_data),
                                                 disabled=dis_h_true
                                                 )
                h_sample_name = st.text_input("Please name H sample",
                                              h_sample_selected
                                              )
                st.write("Define H sample is: ",
                         h_sample_selected
                         )
                st.write(":red[To be:]",
                         h_sample_name
                         )
                df = df.replace({h_sample_selected: h_sample_name})

        with tab_i:
            dis_i_true = True
            dis_i = st.checkbox('Have I sample')
            if dis_i:
                dis_i_true = False
                i_sample_selected = st.selectbox("Which sample define to be I sample name",
                                                 set(sam_data),
                                                 disabled=dis_i_true
                                                 )
                i_sample_name = st.text_input("Please name I sample",
                                              i_sample_selected
                                              )
                st.write("Define I sample is: ",
                         i_sample_selected
                         )
                st.write(":red[To be:]",
                         i_sample_name
                         )
                df = df.replace({i_sample_selected: i_sample_name})

        with tab_j:
            dis_j_true = True
            dis_j = st.checkbox('Have J sample')
            if dis_j:
                dis_j_true = False
                j_sample_selected = st.selectbox("which sample define to be J sample",
                                                 set(sam_data),
                                                 disabled=dis_j_true
                                                 )
                j_sample_name = st.text_input("Please name J sample",
                                              j_sample_selected
                                              )
                st.write("Define J sample is: ",
                         j_sample_selected
                         )
                st.write(":red[To be:]",
                         j_sample_name
                         )
                df = df.replace({j_sample_selected: j_sample_name})

        re_table1, re_table2 = st.columns([0.5, 0.5])
        with re_table1:
            st.write("##### Table now!! #####")
            st.write(df)
        with re_table2:
            st.write("##### Group of sample now!! #####")
            result_replace = df.groupby([sam_select]).count()
            st.dataframe(result_replace)
            sam_count = df.value_counts(sam_select)
    st.sidebar.write(sam_count)
    st.sidebar.write(com_data)
    st.markdown('---')


def intro():
    st.write("##### :red[:warning: Don't forget to prepare your data !! ] #####")


def data_analytic():
    ##|||||||||||||||||||||||||||||||| DATA ANALYTICS SIDEBAR VARIABLES SETTING |||||||||||||||||||||||||||||||||||

    st.write("## DATA ANALYTIC SECTION ##")
    st.sidebar.write("### Select variables to analysis  ###")

    x_selected_vars = st.sidebar.multiselect("Select X (Independent variables) variables",
                                             variables
                                             )
    x_data = pd.DataFrame(df,
                          columns=x_selected_vars
                          )
    y_selected_vars = st.sidebar.multiselect("Select Y (lab result/ Dependent variables) variables",
                                             variables
                                             )
    y_data = pd.DataFrame(df,
                          columns=y_selected_vars
                          )

    ##|||||||||||||||||||||||||||||||||||||| SELECT VARIABLES TO ANALYTIC ||||||||||||||||||||||||||||||||||||||||||

    st.write("#### Variables setting ####")
    tap_op1, tap_op2 = st.columns((0.5, 0.5))

    with tap_op1:
        option_xv = st.selectbox("Select your X variable to show in Histogram",
                                 x_selected_vars
                                 )
        st.write('X variables selected:',
                 option_xv
                 )
        st.dataframe(x_data)
    with tap_op2:
        option_yv = st.selectbox("Select your Y variable to show in Histogram",
                                 y_selected_vars
                                 )
        st.write('Y variables selected:',
                 option_yv
                 )
        st.dataframe(y_data)

    st.write("## Colour setting ##")
    colour_op1, colour_op2 = st.columns((0.5, 0.5))
    with colour_op1:
        sample_color_1 = st.selectbox("Choose sample 1 to change colour",
                                      set(df[sam_select]),
                                      )
        selected_color_1 = st.color_picker("Choose a color of sample 1: " + str(sample_color_1), "#ff5733")
        sample_color_2 = st.selectbox("Choose sample 2 to change colour",
                                      set(df[sam_select]),
                                      )
        selected_color_2 = st.color_picker("Choose a color of sample 2: " + str(sample_color_2), "#ff5733")
        sample_color_3 = st.selectbox("Choose sample 3 to change colour",
                                      set(df[sam_select]),
                                      )
        selected_color_3 = st.color_picker("Choose a color of sample 3: " + str(sample_color_3), "#ff5733")
        sample_color_4 = st.selectbox("Choose sample 4 to change colour",
                                      set(df[sam_select]),
                                      )
        selected_color_4 = st.color_picker("Choose a color of sample 4: " + str(sample_color_4), "#ff5733")
        sample_color_5 = st.selectbox("Choose sample 5 to change colour",
                                      set(df[sam_select]),
                                      )
        selected_color_5 = st.color_picker("Choose a color of sample 5: " + str(sample_color_5), "#ff5733")

    with colour_op2:
        colour_more_true = True
        colour_more = st.checkbox('If more than 5 sample')
        if colour_more:
            colour_more_true = False
            sample_color_6 = st.selectbox("Choose sample 6 to change colour",
                                          set(df[sam_select]),
                                          )
            selected_color_6 = st.color_picker("Choose a color of sample 6: " + str(sample_color_6), "#ff5733")
            sample_color_7 = st.selectbox("Choose sample 7 to change colour",
                                          set(df[sam_select]),
                                          )
            selected_color_7 = st.color_picker("Choose a color of sample 7: " + str(sample_color_7), "#ff5733")
            sample_color_8 = st.selectbox("Choose sample 8 to change colour",
                                          set(df[sam_select]),
                                          )
            selected_color_8 = st.color_picker("Choose a color of sample 8: " + str(sample_color_8), "#ff5733")
            sample_color_9 = st.selectbox("Choose sample 9 to change colour",
                                          set(df[sam_select]),
                                          )
            selected_color_9 = st.color_picker("Choose a color of sample 9: " + str(sample_color_9), "#ff5733")
            sample_color_10 = st.selectbox("Choose sample 10 to change colour",
                                           set(df[sam_select]),
                                           )
            selected_color_10 = st.color_picker("Choose a color of sample 10: " + str(sample_color_10), "#ff5733")

    color_map = {sample_color_1: selected_color_1,
                 sample_color_2: selected_color_2,
                 sample_color_3: selected_color_3,
                 sample_color_4: selected_color_4,
                 sample_color_5: selected_color_5,
                 }
    if colour_more:
        colour_more_true = False
        color_map = {sample_color_6: selected_color_6,
                     sample_color_7: selected_color_7,
                     sample_color_8: selected_color_8,
                     sample_color_9: selected_color_9,
                     sample_color_10: selected_color_10,
                     }

    def histogram_data():

        ###|||||||||||||||||||||||||||||||||||||| HISTOGRAM ANALYTICS ||||||||||||||||||||||||||||||||||||||||||

        import plotly.express as px
        from sklearn.decomposition import PCA
        st.markdown('---')
        st.write("### HISTOGRAM SECTION ###")
        x_axis, y_axis = st.columns([1, 1], gap='medium')

        with x_axis:

            st.write("#### Histogram X variable of result data ####")
            fig_hisx = px.histogram(x_data,
                                    x=option_xv,
                                    hover_data=x_data.columns,
                                    title="Histrogram of "
                                          + str(option_xv),
                                    color=df[sam_select],
                                    color_discrete_map=color_map,
                                    width=500
                                    ).update_xaxes(categoryorder='total descending')  # *

            fig_hisx_max = px.histogram(df,
                                        x=option_xv,
                                        color=sam_data,
                                        histfunc='max',
                                        hover_data=df.columns,
                                        title="Histrogram of "
                                              + str(option_xv)
                                              + str(" max data"),
                                        color_discrete_map=color_map,
                                        width=500
                                        ).update_xaxes(categoryorder='total descending')  #

            fig_hisx_min = px.histogram(df,
                                        x=option_xv,
                                        color=sam_data,
                                        histfunc='min',
                                        hover_data=df.columns,
                                        title="Histrogram of "
                                              + str(option_xv)
                                              + str(" min data"),
                                        color_discrete_map=color_map,
                                        width=500
                                        ).update_xaxes(categoryorder='total descending')  #

            st.plotly_chart(fig_hisx)
            st.plotly_chart(fig_hisx_max)
            st.plotly_chart(fig_hisx_min)

        with y_axis:
            st.write("#### Histogram Y variable of result data ####")
            fig_hisy = px.histogram(y_data,
                                    x=option_yv,
                                    hover_data=y_data.columns,
                                    title="Histrogram of "
                                          + str(option_yv),
                                    color=df[sam_select],
                                    color_discrete_map=color_map,
                                    width=500
                                    ).update_xaxes(categoryorder='total descending')

            fig_hisy_max = px.histogram(df,
                                        x=option_yv,
                                        color=sam_data,
                                        histfunc='max',
                                        hover_data=df.columns,
                                        title="Histrogram of "
                                              + str(option_yv)
                                              + str(" max data"),
                                        color_discrete_map=color_map,
                                        width=500
                                        ).update_xaxes(categoryorder='total descending')  #

            fig_hisy_min = px.histogram(df,
                                        x=option_yv,
                                        color=sam_data,
                                        # color_discrete_map=color_map,
                                        histfunc='min',
                                        hover_data=df.columns,
                                        title="Histrogram of "
                                              + str(option_yv)
                                              + str(" min data"),
                                        color_discrete_map=color_map,
                                        width=500
                                        ).update_xaxes(categoryorder='total descending')  #
            st.plotly_chart(fig_hisy)
            st.plotly_chart(fig_hisy_max)
            st.plotly_chart(fig_hisy_min)

        st.markdown('---')
        st.write("### X-Y axis compair scatter and histogram plot ###")
        z_axis = True
        z_h = st.checkbox('Have Z variable')
        if z_h:
            z_axis = False
            option_zv = st.selectbox("Select your Z variable to show in Histogram",
                                     variables
                                     )
            st.write('Z variables selected:',
                     option_zv
                     )
            st.dataframe(x_data)
        try:
            fig_his3_scatterz = px.scatter(df,
                                           x=option_xv,
                                           y=option_yv,
                                           facet_col=option_zv,
                                           color=df[sam_select],
                                           symbol=df[sam_select],
                                           color_discrete_map=color_map,
                                           marginal_x="box",
                                           #marginal_y="rug",
                                           title="Scatter plot between "
                                                 + str(option_xv)
                                                 + str(" and ")
                                                 + str(option_yv)
                                                 + str(" data add size bubble of ")
                                                 + str(option_zv),
                                           )
            st.plotly_chart(fig_his3_scatterz)
        except:
            pass
        fig_his3_scatter = px.scatter(df,
                                      x=option_xv,
                                      y=option_yv,
                                      template="simple_white",
                                      color=df[sam_select],
                                      symbol=df[sam_select],
                                      color_discrete_map=color_map,
                                      #marginal_x="rug",
                                      marginal_y="box",
                                      title="Scatter plot between "
                                            + str(option_xv)
                                            + str(" and ")
                                            + str(option_yv)
                                            + str(" data "),
                                      )
        fig_his3_scatter.update_layout(font=dict(
                                            family="Times New Roman",
                                            size=12,
                                            color="Black"
                                                )
                                          )
        st.plotly_chart(fig_his3_scatter)
        if st.button("Full fig_his3_scatter"):
            fig_his3_scatter.show()

        #fig_his3_avg = px.histogram(df,
        #                            x=option_xv,
        #                            y=option_yv,
        #                            facet_col=option_zv,
        #                            color=df[sam_select],
        #                            color_discrete_map=color_map,
        #                            histfunc='avg',
        #                            hover_data=df.columns,
        #                            title="Histrogram of "
        #                                  + str(option_yv)
        #                                  + str(" average data"),
        #                            ).update_xaxes(categoryorder='total descending')

        fig_his3_avg_over = px.histogram(df,
                                         x=option_xv,
                                         y=option_yv,
                                         color=df[sam_select],
                                         color_discrete_map=color_map,
                                         marginal="box",
                                         barmode='overlay',
                                         histfunc='avg',
                                         range_x=[0.5,25.0],
                                         hover_data=df.columns,
                                         title="Histrogram of "
                                               + str(option_yv)
                                               + str(" average overlay data"),
                                         ).update_xaxes(categoryorder='total descending')
        st.plotly_chart(fig_his3_avg_over)
        if st.button("Full fig_his3_avg_over"):
            fig_his3_avg_over.show()
        # *"Update_xaxes when you wish the most total all left you should use 'total descending' if contra
        # you should use 'total ascending' and when you catagory alphabet you should use 'catagory as/descending'"

        ##PCA
        import numpy as np
        from sklearn.impute import KNNImputer
        import matplotlib.pyplot as plt
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.preprocessing import StandardScaler

        # Example data (replace with your dataset)
        dfvip = pd.crosstab(index=df[sam_select],
                            columns=df[option_xv],
                            values=df[option_yv],
                            aggfunc='first'
                            ).fillna(0)
        st.write(dfvip)

        #imputer = KNNImputer(n_neighbors=5, metric='nan_euclidean')  # Use 5 nearest neighbors
        #dfvip2 = imputer.fit_transform(dfvip)
        #dfvip3 = pd.DataFrame(dfvip2,
        #                      columns=dfvip.columns,
        #                      index=dfvip.index
        #                      )
        #st.write(dfvip3)

        x_vip = dfvip.iloc[:, :-1]  # Features (all columns except the last)
        y_vip = dfvip.iloc[:, -1]  # Response (last column)

        # Standardize the data (mean=0, variance=1)
        scaler = StandardScaler()
        X_std = scaler.fit_transform(x_vip)
        y_std = (y_vip - np.mean(y_vip)) / np.std(y_vip)  # Standardize y if needed

        # Perform PLS Regression
        n_components = 2  # Number of PLS components
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_std, y_std)
        pls_plot = px.scatter(pls.x_weights_)
        st.plotly_chart(pls_plot)
        # Calculate VIP scores
        def calculate_vip_scores(pls):
            t = pls.x_scores_  # X scores (latent variables)
            w = pls.x_weights_  # X weights
            q = pls.y_loadings_  # Y loadings
            p = x_vip.shape[1]  # Number of variables
            ssy = np.sum(np.square(t @ q.T), axis=0)  # SSY for each component
            vip = np.sqrt(p * np.sum(ssy * np.square(w), axis=1) / np.sum(ssy))
            return vip

        # Calculate VIP scores
        vip_scores = calculate_vip_scores(pls)

        # Create a DataFrame for VIP scores
        vip_df = pd.DataFrame({
            'Variable': x_vip.columns,
            'VIP_Score': vip_scores
        })

        # Sort by VIP scores in descending order
        vip_df = vip_df.sort_values(by='VIP_Score', ascending=False)

        st.write(vip_df)


        vip_scat = px.scatter(vip_df,
                                x="VIP_Score",
                                y="Variable",
                                template="simple_white",
                                title="VIP score of "
                                        + str(option_xv),
                                #marginal_y=vi,
                                color_discrete_sequence=["black", "grey", "white"]
                                 ).update_yaxes(categoryorder='total ascending')

        vip_scat.update_layout( font=dict(
                                        family="Times New Roman",
                                        size=10,
                                        color="Black"
                                         ),
                                xaxis = dict(
                                        title=dict(
                                            text="VIP score"
                                                    )
                                            ),
                                yaxis = dict(
                                    title=dict(
                                        text=option_xv
                                                )
                                            )
                                )
        st.plotly_chart(vip_scat)
        if st.button("Full vip_scat"):
            vip_scat.show()

        vip_dfB = vip_df.head(10)

        vip_his =px.histogram(vip_dfB,
                               x="VIP_Score",
                               y="Variable",
                               color="Variable",
                               pattern_shape="Variable",
                               #pattern_shape_sequence = [".", "x", "+"],
                               template="simple_white",
                               title="Top 10 VIP score of "
                                            +str(option_xv),
                               color_discrete_sequence=["black", "gray", "darkblue", "crimson"]
                               ).update_yaxes(categoryorder='total ascending')

        vip_his.update_layout( font=dict(
                                        family="Times New Roman, bold",
                                        size=12,
                                        color="black"
                                         ),
                                xaxis = dict(
                                        title=dict(
                                            text="VIP score"
                                                    )
                                            ),
                                yaxis = dict(
                                    title=dict(
                                        text=option_xv
                                                )
                                            )
                                )
        st.plotly_chart(vip_his)
        if st.button("Full vip_his"):
            vip_his.show()
        X = df[[option_xv, option_yv, option_zv]]

        #X = np.log1p(X)
        #X = X.div(X.sum(axis=1), axis=0)
        # Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(X)

        # Apply PCA
        n_components = 3
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(data_scaled)

        # Calculate total explained variance
        total_var = pca.explained_variance_ratio_.sum() - 1

        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        # Create labels for principal components

        labels = {str(i): f"PC{i + 1}" for i in range(n_components)}
        st.header("Data processing")
        st.subheader("Data input")
        st.dataframe(X)
        st.subheader("PCA components")
        st.dataframe(components)
        aa = pca.explained_variance_ratio_
        explained_var_df = pd.DataFrame({
            "Component": [f"PC{i + 1}" for i in range(n_components)],
            "Explained Variance": aa
        })
        st.subheader("Explained Variance Ratio")
        st.dataframe(explained_var_df)
        pca1_2, pca2_3 = st.columns([0.5, 0.5])

        with pca1_2:
            # Create a Plotly scatter plot (PC1 - PC2)
            fig_pca1 = px.scatter(
                components,
                x=0,
                y=1,
                color=df[sam_select],  # Ensure sam_select is a valid column name
                text=df[com_select].drop_duplicates(),
                color_discrete_map=color_map,
                labels=labels,
                title="PCA1 & PCA2",
            )
            # Update the text position and style if sample names are shown
            fig_pca1.update_traces(
                textposition="top center",  # Position of the sample names
                textfont=dict(size=9, color="black")  # Font size and color
            )
            # Update axis labels
            fig_pca1.update_layout(
                xaxis_title=f"Principal Component 1 (PC1: {aa[0]:.2f}%)",
                yaxis_title=f"Principal Component 2 (PC2: {aa[1]:.2f}%)"
            )

            for i, feature in enumerate(X.columns):
                fig_pca1.add_annotation(
                    ax=0, ay=0,
                    axref="x", ayref="y",
                    x=loadings[i, 0],
                    y=loadings[i, 1],
                    showarrow=True,
                    arrowsize=2,
                    arrowhead=2,
                    xanchor="right",
                    yanchor="top"
                )
                fig_pca1.add_annotation(
                    x=loadings[i, 0],
                    y=loadings[i, 1],
                    ax=0, ay=0,
                    xanchor="center",
                    yanchor="bottom",
                    text=feature,
                    yshift=5,
                )

            st.plotly_chart(fig_pca1)

        with pca2_3:
            # Create a Plotly scatter plot (PC2 - PC3)
            fig_pca2 = px.scatter(
                components,
                x=1,
                y=2,
                color=df[sam_select],  # Ensure sam_select is a valid column name
                color_discrete_map=color_map,
                labels=labels,
                title="PCA2 & PCA3"
            )

            # Update axis labels
            fig_pca2.update_layout(
                xaxis_title=f"Principal Component 2 (PC2: {aa[1]:.2f}%)",
                yaxis_title=f"Principal Component 3 (PC3: {aa[2]:.2f}%)"
            )


            for i, feature in enumerate(X.columns):
                fig_pca2.add_annotation(
                    ax=0, ay=0,
                    axref="x", ayref="y",
                    x=loadings[i, 1],
                    y=loadings[i, 2],
                    showarrow=True,
                    arrowsize=2,
                    arrowhead=2,
                    xanchor="right",
                    yanchor="top"
                )
                fig_pca2.add_annotation(
                    x=loadings[i, 1],
                    y=loadings[i, 2],
                    ax=0, ay=0,
                    xanchor="center",
                    yanchor="bottom",
                    text=feature,
                    yshift=5,
                )
            st.plotly_chart(fig_pca2)


        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        fig_cumulative = px.area(
        x=range(1, n_components + 1),
        y=cumulative_variance,
        labels={"x": "Number of Components", "y": "Cumulative Explained Variance"},
        title="Cumulative Explained Variance by PCA Components"
        )
        st.subheader("Cumulative Explained Variance")
        st.plotly_chart(fig_cumulative)



            # Step 5: Visualize the PCA Results

    def correlation_data():
        import seaborn as sns
        import scipy.stats as stats
        import matplotlib.pyplot as plt

        ###|||||||||||||||||||||||||||||||||||||| STAT CORRELATION ANALYTICS ||||||||||||||||||||||||||||||||||||||||||
        st.markdown('---')
        st.write("### CORRELATION SECTION (use X variable and Y variable) ###")
        st.write("#### don't forget to set X, Y variable ####")
        cor = pd.crosstab(df[option_yv],
                          df[option_xv]
                          )
        count, z = st.columns([0.5, 0.5])
        with count:
            st.write("Count Data")
            st.write(cor)
        with z:
            st.write("Z score table between: " + str(option_xv) + " & " + str(option_yv))
            cor_z = stats.zscore(cor,
                                 axis=1,
                                 nan_policy='omit')
            st.dataframe(cor_z)

        def cor_data():
            method_cor = st.selectbox("What method to do correlation",
                                      ('pearson', 'kendall', 'spearman')
                                      )
            cor2 = cor.corr(method=method_cor)
            cc = pd.DataFrame(cor2)
            st.write(cc)
            res1, res2, res3 = st.columns([0.3, 0.3, 0.3])
            with res1:
                st.write(
                    "Table score of: " + str(method_cor) + " model, between " + str(option_xv) + " & " + str(option_yv))
                st.dataframe(cor2)
            with res2:
                st.write(
                    "Heat-table score of: " + str(method_cor) + " model, between " + str(option_xv) + " & " + str(
                        option_yv))
                cor_heat = plt.figure()
                sns.heatmap(cc, annot=True, cmap="RdBu_r", vmin=-1, vmax=1)
                st.pyplot(cor_heat)
            with res3:
                fig_cor = sns.clustermap(cc,
                                         annot=True,
                                         annot_kws={"size": 20},
                                         row_cluster=True,
                                         vmin=-1,
                                         vmax=1,
                                         center=0,
                                         cmap="RdBu_r"
                                         )
                st.pyplot(fig_cor)

            st.write("Correlation Matrix of " + str(method_cor) + " model, between " + str(option_xv) + " & " + str(
                option_yv))
            cor2.apply(lambda x: x.factorize()[0]).corr()
            cors = sns.pairplot(cor2)
            st.pyplot(cors)

        ###|||||||||||||||||||||||||||||||||||| ANOVA ||||||||||||||||||||||||||||||||||||||||||||||||||||||
        def ano_data():
            ano1, ano2, ano3, ano4 = st.columns([0.2, 0.2, 0.2, 0.2])
            ano5, ano6, ano7, ano8 = st.columns([0.2, 0.2, 0.2, 0.2])

            with ano1:
                dis_ano1_check = True
                A1 = []
                dis_ano1 = st.checkbox('Have 1st variable')
                if dis_ano1:
                    dis_ano1_check = False
                    col1 = st.selectbox("Select 1st variable:",
                                        df.columns,
                                        disabled=dis_ano1_check
                                        )

                    A1 = df[col1]
                    st.write(A1)

            with ano2:
                dis_ano2_check = True
                A2 = []
                dis_ano2 = st.checkbox('Have 2nd variable')
                if dis_ano2:
                    dis_ano2_check = False
                    col2 = st.selectbox("Select 2nd variable:",
                                        df.columns,
                                        disabled=dis_ano2_check
                                        )
                    A2 = df[col2]
                    st.write(A2)
                    fvalue1, pvalue1 = stats.kruskal(A1, A2)
                    st.write("F Value: " + str(fvalue1))
                    st.write("P Value: " + str(pvalue1))

            with ano3:
                dis_ano3_check = True
                A3 = []
                dis_ano3 = st.checkbox('Have 3rd variable')
                if dis_ano3:
                    dis_ano3_check = False
                    col3 = st.selectbox("Select 3rd variable:",
                                        df.columns,
                                        disabled=dis_ano3_check
                                        )
                    A3 = df[col3]
                    st.write(A3)
                    fvalue2, pvalue2 = stats.kruskal(A1, A2, A3)
                    st.write("F Value: " + str(fvalue2))
                    st.write("P Value: " + str(pvalue2))

            with ano4:
                dis_ano4_check = True
                A4 = []
                dis_ano4 = st.checkbox('Have 4th variable')
                if dis_ano4:
                    dis_ano4_check = False
                    col4 = st.selectbox("Select 4th variable:",
                                        df.columns,
                                        disabled=dis_ano4_check
                                        )
                    A4 = df[col4]
                    st.write(A4)
                    fvalue3, pvalue3 = stats.kruskal(A1, A2, A3, A4)
                    st.write("F Value: " + str(fvalue3))
                    st.write("P Value: " + str(pvalue3))

            with ano5:
                dis_ano5_check = True
                A5 = []
                dis_ano5 = st.checkbox('Select 5th variable:')
                if dis_ano5:
                    dis_ano5_check = False
                    col5 = st.selectbox("Columns:",
                                        df.columns,
                                        disabled=dis_ano5_check
                                        )
                    A5 = df[col5]
                    st.write(A5)
                    fvalue4, pvalue4 = stats.kruskal(A1, A2, A3, A4, A5)
                    st.write("F Value: " + str(fvalue4))
                    st.write("P Value: " + str(pvalue4))

            with ano6:
                dis_ano6_check = True
                A6 = []
                dis_ano6 = st.checkbox('Have 6th variable')
                if dis_ano6:
                    dis_ano6_check = False
                    col6 = st.selectbox("Select 6th variable:",
                                        df.columns,
                                        disabled=dis_ano6_check
                                        )
                    A6 = df[col6]
                    st.write(A6)
                    fvalue5, pvalue5 = stats.kruskal(A1, A2, A3, A4, A5, A6)
                    st.write("F Value: " + str(fvalue5))
                    st.write("P Value: " + str(pvalue5))

            with ano7:
                dis_ano7_check = True
                A7 = []
                dis_ano7 = st.checkbox('Have 7th variable')
                if dis_ano7:
                    dis_ano7_check = False
                    col7 = st.selectbox("Select 7th variable:",
                                        df.columns,
                                        disabled=dis_ano7_check
                                        )
                    A7 = df[col7]
                    st.write(A7)
                    fvalue6, pvalue6 = stats.kruskal(A1, A2, A3, A4, A5, A6, A7)
                    st.write("F Value: " + str(fvalue6))
                    st.write("P Value: " + str(pvalue6))

            with ano8:
                dis_ano8_check = True
                A8 = []
                dis_ano8 = st.checkbox('Have 8 variable')
                if dis_ano8:
                    dis_ano8_check = False
                    col8 = st.selectbox("Select 8th variable:",
                                        df.columns,
                                        disabled=dis_ano8_check
                                        )
                    A8 = df[col8]
                    st.write(A8)
                    fvalue7, pvalue7 = stats.kruskal(A1, A2, A3, A4, A5, A6, A7, A8)
                    st.write("F Value: " + str(fvalue7))
                    st.write("P Value: " + str(pvalue7))

        pagecorr_names_to_funcs = {
            "CORRELATION STAT": cor_data,
            "ANOVA": ano_data
        }

        correlation_name = st.selectbox("Choose analytic plan",
                                        pagecorr_names_to_funcs.keys()
                                        )
        pagecorr_names_to_funcs[correlation_name]()

    def scatter_data():
        ###|||||||||||||||||||||||||||||||||||||| SCATTER PLOT ANALYTICS ||||||||||||||||||||||||||||||||||||||||||
        import plotly.express as px
        st.markdown('---')
        st.write("### SCATTER PLOT SECTION ###")
        z_dot = True
        z_s = st.checkbox('Have Z variable')
        if z_s:
            z_dot = False
            option_z = st.selectbox("Select ot size variables",
                                variables
                                )
        try:

            fig_scat0 = px.scatter(df,
                               x=option_xv,
                               y=option_yv,
                               size=option_z,
                               color=sam_select,
                               symbol=sam_select,
                               text=option_xv,
                               color_discrete_map=color_map,
                               #marginal_x='box',
                               marginal_y='box',
                               title="Scatter of "
                                     + str(option_xv)
                                     + str(" and ")
                                     + str(option_yv),
                               )
            fig_scat0.update_layout(font_family="Times New Roman",
                                    font_color="black")
            st.plotly_chart(fig_scat0)


        except:
            pass

        fig_scat1 = px.scatter(df,
                               x=option_xv,
                               y=option_yv,
                               #size=option_z,
                               color=sam_select,
                               symbol=sam_select,
                               text=option_xv,
                               color_discrete_map=color_map,
                               #marginal_x='box',
                               marginal_y='box',
                               title="Scatter of "
                                     + str(option_xv)
                                     + str(" and ")
                                     + str(option_yv),
                               )
        fig_scat1.update_layout(font_family="Times New Roman",
                                font_color="black")
        st.plotly_chart(fig_scat1)

        col_option1 = st.selectbox("Select spread column variables",
                                                 variables
                                                 )
        #col_option2 = st.selectbox("Select spread row variables",
        #                                         variables
        #                                         )
        fig_scat2 = px.scatter(df,
                               x=option_xv,
                               y=option_yv,
                               color=sam_select,
                               color_discrete_map=color_map,
                               facet_row=col_option1,
                               text=option_xv,
                               #facet_row=col_option2,
                               marginal_x='box',
                               marginal_y='box',
                               title="Scatter of "
                                     + str(option_xv)
                                     + str(" and ")
                                     + str(option_yv),
                               )
        st.plotly_chart(fig_scat2)

    def heat_data():
        ###|||||||||||||||||||||||||||||||||||||| HEAT-MAP FROM MD |||||||||||||||||||||||||||||||||
        import matplotlib.pyplot as plt
        import seaborn as sns
        st.write("### HEAT-MAP SECTION ###")
        df.apply(lambda x: x.factorize()[0]).corr()
        option_zv = st.selectbox("Select your Z variable to show in Histogram",
                                 variables
                                 )
        st.write('Z variables selected:',
                 option_zv
                 )
        h_data = pd.crosstab(
                             index=df[option_xv],
                             columns=df[option_yv],
                             values=df[option_zv],
                             aggfunc='sum',
                             dropna=True
                             ).fillna(0)

        st.write(h_data)
        fig_hm1 = plt.figure()
        annot_check = False
        annot = st.checkbox("Show annot")
        if annot:
            annot_check = True
        sns.heatmap(h_data,
                    annot=annot_check,
                    cmap="Reds",
                    center=0,
                    yticklabels=1,
                    )
        row_heat_data = st.slider("How much row width",
                                  1,
                                  10,
                                  2
                                  )
        col_heat_data = st.slider("How much column width",
                                  1,
                                  10,
                                  5
                                  )
        fig_hm1.subplots_adjust(top=col_heat_data,
                                bottom=row_heat_data,
                                #hspace=2,
                                )
        st.pyplot(fig_hm1)

        fig_hm2 = plt.figure()
        row_clus_data = st.slider("How much row width",
                                  1,
                                  20,
                                  10
                                  )
        col_clus_data = st.slider("How much column width",
                                  1,
                                  20,
                                  10
                                  )
        fig_hm2 = sns.clustermap(h_data,
                                 annot=annot_check,
                                 cmap="Reds",
                                 center=2,
                                 annot_kws={"size": 8, "family": "Times New Roman"},
                                 yticklabels=1,
                                 figsize=(col_clus_data,row_clus_data),
                                 )
        fig_hm2.tick_params(axis='both',
                            labelsize=10,
                            labelfontfamily="Times New Roman"
                            )
        st.pyplot(fig_hm2)


    def venn_data():
        ###|||||||||||||||||||||||||||||||||||||| VENN-DIAGRAM ||||||||||||||||||||||||||||||||||||||||||
        import matplotlib.pyplot as plt
        from matplotlib_venn import venn3
        from matplotlib_venn import venn2

        st.markdown('---')
        st.write("### VENN-DIAGRAM SECTION ###")
        st.write("VARIABLE")
        option_yv_venn = st.selectbox("CHOICE Y VARIABLE TO COMPAIR WITH SAMPLES",
                                      y_selected_vars
                                      )
        st.write('YOU SELECTED: ',
                 option_yv_venn
                 )

        sam_data2 = df[sam_select]
        st.write("SELECT SAMPLE TO BE ANALYTIC")
        st.dataframe(df)
        st.write("## RESULT DATA VENN-DIAGRAM ANALYTIC ##")
        ven_a, ven_b = st.columns((0.5, 0.5),
                                  gap="large"
                                  )

        ##|||||||||||||||||||||||||||||||||||||| VENN-A ||||||||||||||||||||||||||||||||||||||||||

        with ven_a:
            st.write("Venn A Section")
            acr1_select_sample = st.selectbox("Select sample A1", options=set(sam_data2))
            acr1_selected = set(df.loc[df[sam_select] == acr1_select_sample][option_yv_venn])
            acr1_color_selected = st.color_picker("Choose a color in sample A1", "#ff5733")
            st.dataframe(acr1_selected)

            acr2_select_sample = st.selectbox("Select sample A2", options=set(sam_data2))
            acr2_selected = set(df.loc[df[sam_select] == acr2_select_sample][option_yv_venn])
            acr2_color_selected = st.color_picker("Choose a color in sample A2", "#ff5733")
            st.dataframe(acr2_selected)

            #acr3_select_sample = st.selectbox("Select sample A3", options=set(sam_data2))
            #acr3_selected = set(df.loc[df[sam_select] == acr3_select_sample][option_yv_venn])
            #acr3_color_selected = st.color_picker("Choose a color in sample A3", "#ff5733")
            #st.dataframe(acr3_selected)

            ##|||||||||||||||||||||||||||||||||| FIGURE VENN-A ||||||||||||||||||||||||||||||||||||||

            st.write(":red[(A)]" + str(" VENN-DIAGRAM DATA RESULT"))
            fig_av, ax = plt.subplots(1, 1)
            va = venn2([acr1_selected,
                        acr2_selected,
                        ],
                        #acr3_selected],
                       (acr1_select_sample,
                        acr2_select_sample,
                        ),
                        #acr3_select_sample),
                       ax=ax
                       )
            try:
                va.get_patch_by_id('100').set_color(acr1_color_selected)
            except:
                pass
            try:
                va.get_patch_by_id('010').set_color(acr2_color_selected)
            except:
                pass
            #try:
            #    va.get_patch_by_id('001').set_color(acr3_color_selected)
            #except:
            #    pass

            va.get_label_by_id('A').set_text(acr1_select_sample)
            va.get_label_by_id('B').set_text(acr2_select_sample)
            #va.get_label_by_id('C').set_text(acr3_select_sample)
            st.pyplot(fig_av)

            venn_diagram_a1, venn_diagram_a2 = st.columns((0.7, 0.3)
                                                          )
            with venn_diagram_a1:

                ##|||||||||||||||||||||||||||||||||| TABLE SET VENN-A ||||||||||||||||||||||||||||||||||||||

                a_sam_blank = set()
                a_inner_all = set.intersection(acr1_selected,
                                               acr2_selected,
                                               #acr3_selected
                                               )
                a_outer_sam1 = set(acr1_selected - acr2_selected
                                   #- acr3_selected
                                   )
                a_outer_sam2 = set(acr2_selected - acr1_selected
                                   #- #acr3_selected
                                   )
                #a_outer_sam3 = set(acr3_selected - acr2_selected - acr1_selected)
                a_union_sam1_2 = set.union(acr1_selected,
                                           acr2_selected)
                #a_union_sam1_3 = set.union(acr1_selected,
                                           #acr3_selected)
                #a_union_sam2_3 = set.union(acr2_selected,
                                           #acr3_selected)
                a_inner_sam1_2 = set.intersection(acr1_selected,
                                                  acr2_selected)
                #a_inner_sam1_3 = set.intersection(acr1_selected,
                                                  #acr3_selected)
                #a_inner_sam2_3 = set.intersection(acr2_selected,
                                                  #acr3_selected)
                a_inner_sam1_2_ex_inner = (set.intersection(acr1_selected,
                                                            acr2_selected)) - (set.intersection(acr1_selected,
                                                                                                acr2_selected,
                                                                                                #acr3_selected
                                                                                                )
                                                                               )
                #a_inner_sam1_3_ex_inner = (set.intersection(acr1_selected,
                #                                           acr3_selected)) - (set.intersection(acr1_selected,
                #                                                                               acr2_selected,
                #                                                                               acr3_selected))
                #a_inner_sam2_3_ex_inner = (set.intersection(acr2_selected,
                #                                           acr3_selected)) - (set.intersection(acr1_selected,
                #                                                                               acr2_selected,
                #                                                                               acr3_selected))
                choice_va = {
                    'VennA_Blank': a_sam_blank,
                    'VennA_Inner all circle': a_inner_all,
                    'VennA_Outer sample 1': a_outer_sam1,
                    'VennA_Outer sample 2 ': a_outer_sam2,
                    #'VennA_Outer sample 3': a_outer_sam3,
                    'VennA_Circle sample 1 and 2 (1u2)': a_union_sam1_2,
                    #'VennA_Circle sample 1 and 3 (1u3)': a_union_sam1_3,
                    #'VennA_Circle sample 2 and 3 (2u3)': a_union_sam2_3,
                    'VennA_Inner circle 1 and 2': a_inner_sam1_2,
                    #'VennA_Inner circle 1 and 3': a_inner_sam1_3,
                    #'VennA_Inner circle 2 and 3': a_inner_sam2_3,
                    'VennA_Inner circle 1 and 2 exclude center': a_inner_sam1_2_ex_inner,
                    #'VennA_Inner circle 1 and 3 exclude center': a_inner_sam1_3_ex_inner,
                    #'VennA_Inner circle 2 and 3 exclude center': a_inner_sam2_3_ex_inner,
                }
                a_set_list = choice_va.keys()
                a_select_set = st.selectbox("WHAT REGION TO SHOW in VENN A",
                                            options=a_set_list,
                                            )
            with venn_diagram_a2:
                st.dataframe(choice_va[a_select_set])
                a_set_result = pd.DataFrame(choice_va[a_select_set])

                def convert_a_set_result(a_set_result):
                    return a_set_result.to_csv(index=False).encode('utf-8')

                a_csv = convert_a_set_result(a_set_result)

                st.download_button("Download this Venn A set data",
                                   a_csv,
                                   a_select_set,
                                   "text/csv",
                                   key='download-Avenn_csv'
                                   )
        with ven_b:

            ##|||||||||||||||||||||||||||||||||||||| VENN-B ||||||||||||||||||||||||||||||||||||||||||

            st.write("Venn B Section")
            bcr1_select_sample = st.selectbox("Select sample B1", options=set(sam_data2))
            bcr1_selected = set(df.loc[df[sam_select] == bcr1_select_sample][option_yv_venn])
            bcr1_color_selected = st.color_picker("Choose a color in sample B1", "#ff5733")
            st.dataframe(bcr1_selected)

            bcr2_select_sample = st.selectbox("Select sample B2", options=set(sam_data2))
            bcr2_selected = set(df.loc[df[sam_select] == bcr2_select_sample][option_yv_venn])
            bcr2_color_selected = st.color_picker("Choose a color in sample B2", "#ff5733")
            st.dataframe(bcr2_selected)

            bcr3_select_sample = st.selectbox("Select sample B3", options=set(sam_data2))
            bcr3_selected = set(df.loc[df[sam_select] == bcr3_select_sample][option_yv_venn])
            bcr3_color_selected = st.color_picker("Choose a color in sample B3", "#ff5733")
            st.dataframe(bcr3_selected)

            ##|||||||||||||||||||||||||||||||||||||| FIGURE VENN-B ||||||||||||||||||||||||||||||||||||||||||

            st.write(":blue[(B)]" + str(" VENN-DIAGRAM DATA RESULT"))
            fig_bv, ax = plt.subplots(1, 1)
            vb = venn3([bcr1_selected,
                        bcr2_selected,
                        bcr3_selected],
                       (bcr1_select_sample, bcr2_select_sample, bcr3_select_sample),
                       ax=ax
                       )
            try:
                vb.get_patch_by_id('100').set_color(bcr1_color_selected)
            except:
                pass
            try:
                vb.get_patch_by_id('010').set_color(bcr2_color_selected)
            except:
                pass
            try:
                vb.get_patch_by_id('001').set_color(bcr3_color_selected)
            except:
                pass

            vb.get_label_by_id('A').set_text(bcr1_select_sample)
            vb.get_label_by_id('B').set_text(bcr2_select_sample)
            vb.get_label_by_id('C').set_text(bcr3_select_sample)
            st.pyplot(fig_bv)

            venn_diagram_b1, venn_diagram_b2 = st.columns((0.7, 0.3)
                                                          )
            with venn_diagram_b1:

                ##|||||||||||||||||||||||||||||||||||||| SET VENN-B ||||||||||||||||||||||||||||||||||||||||||

                b_sam_blank = set()
                b_inner_all = set.intersection(bcr1_selected,
                                               bcr2_selected,
                                               bcr3_selected)
                b_outer_sam1 = set(bcr1_selected - bcr2_selected - bcr3_selected)
                b_outer_sam2 = set(bcr2_selected - bcr1_selected - bcr3_selected)
                b_outer_sam3 = set(bcr3_selected - bcr2_selected - bcr1_selected)
                b_union_sam1_2 = set.union(bcr1_selected,
                                           bcr2_selected)
                b_union_sam1_3 = set.union(bcr1_selected,
                                           bcr3_selected)
                b_union_sam2_3 = set.union(bcr2_selected,
                                           bcr3_selected)
                b_inner_sam1_2 = set.intersection(bcr1_selected,
                                                  bcr2_selected)
                b_inner_sam1_3 = set.intersection(bcr1_selected,
                                                  bcr3_selected)
                b_inner_sam2_3 = set.intersection(bcr2_selected,
                                                  bcr3_selected)
                b_inner_sam1_2_ex_inner = (set.intersection(bcr1_selected,
                                                            bcr2_selected)) - (set.intersection(bcr1_selected,
                                                                                                bcr2_selected,
                                                                                                bcr3_selected))
                b_inner_sam1_3_ex_inner = (set.intersection(bcr1_selected,
                                                            bcr3_selected)) - (set.intersection(bcr1_selected,
                                                                                                bcr2_selected,
                                                                                                bcr3_selected))
                b_inner_sam2_3_ex_inner = (set.intersection(bcr2_selected,
                                                            bcr3_selected)) - (set.intersection(bcr1_selected,
                                                                                                bcr2_selected,
                                                                                                bcr3_selected))

                choice_vb = {
                    'VennB_Blank': b_sam_blank,
                    'VennB_Inner all circle': b_inner_all,
                    'VennB_Outer sample 1': b_outer_sam1,
                    'VennB_Outer sample 2 ': b_outer_sam2,
                    'VennB_Outer sample 3': b_outer_sam3,
                    'VennB_Circle sample 1 and 2 (1u2)': b_union_sam1_2,
                    'VennB_Circle sample 1 and 3 (1u3)': b_union_sam1_3,
                    'VennB_Circle sample 2 and 3 (2u3)': b_union_sam2_3,
                    'VennB_Inner circle 1 and 2': b_inner_sam1_2,
                    'VennB_Inner circle 1 and 3': b_inner_sam1_3,
                    'VennB_Inner circle 2 and 3': b_inner_sam2_3,
                    'VennB_Inner circle 1 and 2 exclude center': b_inner_sam1_2_ex_inner,
                    'VennB_Inner circle 1 and 3 exclude center': b_inner_sam1_3_ex_inner,
                    'VennB_Inner circle 2 and 3 exclude center': b_inner_sam2_3_ex_inner
                }

                b_set_list = choice_vb.keys()
                b_select_set = st.selectbox("WHAT REGION TO SHOW in VENN B",
                                            options=b_set_list,
                                            )
            with venn_diagram_b2:

                st.dataframe(choice_vb[b_select_set])
                b_set_result = pd.DataFrame(choice_vb[b_select_set])

                def convert_b_set_result(b_set_result):
                    return b_set_result.to_csv(index=False).encode('utf-8')

                b_csv = convert_b_set_result(b_set_result)

                st.download_button("Download this Venn B set data",
                                   b_csv,
                                   b_select_set,
                                   "text/csv",
                                   key='download-venn_csv'
                                   )

    ###|||||||||||||||||||||||||||||||||||||| SUB-PAGE SETTING ||||||||||||||||||||||||||||||||||||||||||

    subpage_names_to_funcs = {
        "HISTOGRAM ANALYTIC": histogram_data,
        "CORRELATION ANALYTICS": correlation_data,
        "SCATTER PLOT": scatter_data,
        "HEAT-MAP": heat_data,
        "VENN-DIAGRAM": venn_data
    }

    analytic_name = st.sidebar.selectbox("Choose analytic plan",
                                         subpage_names_to_funcs.keys()
                                         )
    subpage_names_to_funcs[analytic_name]()


def merge_analytic():
    ###|||||||||||||||||||||||||||||||||||||| MERGE DATA |||||||||||||||||||||||||||||||||
    import numpy as np
    import pandas as pd
    ##|||||||||||||||||||||||||||||||||||||| OPEN DATABASE CSV FILE |||||||||||||||||||||||||||||||||

    st.sidebar.write("### :red[Pick your data to be merge] ###")
    uploaded_data = st.sidebar.file_uploader("Choice your merge data in this box",
                                             type=".xlsx"
                                             )
    if uploaded_data:
        df2 = pd.read_excel(uploaded_data)
        st.markdown("#### Data for merge")
        st.dataframe(df2.head(100),
                     hide_index=True
                     )
        merge_variables = df2.columns
        merge_variables = np.array(merge_variables)

        ##|||||||||||||||||||||||||||||||||||||| MERGE DATA (MD) SETTING |||||||||||||||||||||||||||||||||

        st.markdown("## MERGE DATA ###")
        st.write("Don't forget to select variable")

        choice_merge = st.selectbox("Choice column to be merge"
                                    ":red[  (Make sure to named the columns exactly as you want to merge.)]",
                                    merge_variables)
        #choice_merge2 = st.selectbox("Choice column to be merge 2"
        #                            ":red[  (Make sure to named the columns exactly as you want to merge.)]",
        #                            merge_variables)
        choice_how_merge = st.selectbox('HOW TO MERGE THIS DATA', ["inner", "outer", "left", "right"])

        mdf = df.merge(df2,
                       on=[choice_merge],
                       how=choice_how_merge,
                       copy=False,
                       )


        ##|||||||||||||||||||||||||||||||||||||| MERGE RESULT TABLE |||||||||||||||||||||||||||||||||

        st.markdown("## MERGE RESULT ##")
        st.dataframe(mdf)
        def convert_mdf(mdf):
            return mdf.to_csv(index=False).encode('utf-8')

        m_csv = convert_mdf(mdf)

        st.download_button("DOWNLOAD MERGE TABLE",
                           m_csv,
                           "Merge data",
                           "text/csv",
                           key='download-m_csv'
                           )



###|||||||||||||||||||||||||||||||||||| MAIN SUB-PAGE SETTING ||||||||||||||||||||||||||||||||||||||||

page_names_to_funcs = {
                        "—": intro,
                        "DATA ANALYTICS": data_analytic,
                        "MERGE DATA AND ANALYTICS": merge_analytic,
                        }
page_name = st.sidebar.selectbox("Choose what you want to do",
                                 page_names_to_funcs.keys()
                                 )
page_names_to_funcs[page_name]()
